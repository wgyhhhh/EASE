import json
import logging
import os
import queue
import re
import sys
import threading
import uuid
import csv
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

os.environ["EASE_DISABLE_KEY_PROMPT"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT / "Agent"
SCRIPTS_ROOT = AGENT_ROOT / "scripts"
CONFIG_ROOT = AGENT_ROOT / "config"
RUN_PY_PATH = SCRIPTS_ROOT / "run.py"
API_KEYS_PATH = CONFIG_ROOT / "api_keys.yaml"
MODELS_CSV_PATH = CONFIG_ROOT / "available_models.csv"
DEFAULT_INPUT_PATH = AGENT_ROOT / "data" / "input.json"
DEFAULT_OUTPUT_PATH = AGENT_ROOT / "data" / "output.json"

for path in (AGENT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_api_keys() -> dict[str, str]:
    return yaml.safe_load(API_KEYS_PATH.read_text(encoding="utf-8")) or {}


def save_api_keys(api_keys: dict[str, str]) -> None:
    current = load_api_keys()
    current.update(api_keys)
    API_KEYS_PATH.write_text(yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_default_language() -> str:
    content = RUN_PY_PATH.read_text(encoding="utf-8")
    match = re.search(r'os\.environ\["LANGUAGE"\]\s*=\s*"(zh|en)"', content)
    return match.group(1) if match else "zh"


def write_default_language(language: str) -> None:
    content = RUN_PY_PATH.read_text(encoding="utf-8")
    updated = re.sub(
        r'os\.environ\["LANGUAGE"\]\s*=\s*"(zh|en)"',
        f'os.environ["LANGUAGE"] = "{language}"',
        content,
        count=1,
    )
    RUN_PY_PATH.write_text(updated, encoding="utf-8")


def load_models() -> list[dict[str, Any]]:
    models = []
    with open(MODELS_CSV_PATH, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            models.append(
                {
                    "shorthand": row["Shorthand"],
                    "name": row["Name"],
                    "platform": row["Platform"],
                    "input_cost_per_m": float(row["Cost per 1M input tokens"]),
                    "output_cost_per_m": float(row["Cost per 1M output tokens"]),
                }
            )
    return models


class ConfigPayload(BaseModel):
    api_keys: dict[str, str]
    language: str


class RunPayload(BaseModel):
    api_keys: dict[str, str]
    language: str
    input_path: str = str(DEFAULT_INPUT_PATH)
    output_path: str = str(DEFAULT_OUTPUT_PATH)
    model: str = "deepseek_chat"


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    logs: list[str] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    progress_current: int = 0
    progress_total: int = 0
    summary: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "logs": list(self.logs),
                "progress_current": self.progress_current,
                "progress_total": self.progress_total,
                "summary": self.summary,
                "error": self.error,
            }

    def publish(self, event: dict[str, Any]) -> None:
        with self.lock:
            if event["type"] == "log" and event.get("message"):
                self.logs.append(event["message"])
            if event["type"] == "status":
                self.status = event["status"]
            if event["type"] == "progress":
                self.progress_current = event["current"]
                self.progress_total = event["total"]
                self.summary = event["summary"]
            if event["type"] == "complete":
                self.status = "completed"
                self.summary = event["summary"]
            if event["type"] == "error":
                self.status = "failed"
                self.error = event["error"]
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    def add_subscriber(self) -> queue.Queue:
        subscriber = queue.Queue()
        with self.lock:
            self.subscribers.append(subscriber)
        return subscriber

    def remove_subscriber(self, subscriber: queue.Queue) -> None:
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)


class QueueTextStream:
    def __init__(self, job: JobState):
        self.job = job
        self.buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self.buffer += data.replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.job.publish({"type": "log", "message": line})
        return len(data)

    def flush(self) -> None:
        line = self.buffer.strip()
        if line:
            self.job.publish({"type": "log", "message": line})
        self.buffer = ""


class JobLogHandler(logging.Handler):
    def __init__(self, job: JobState):
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record).strip()
        if message:
            self.job.publish({"type": "log", "message": message})


jobs: dict[str, JobState] = {}
app = FastAPI(title="EASE Web UI")


def validate_language(language: str) -> str:
    value = (language or "zh").lower()
    if value not in {"zh", "en"}:
        raise HTTPException(status_code=400, detail="language must be zh or en")
    return value


def run_job(job: JobState, payload: RunPayload) -> None:
    stream = QueueTextStream(job)
    log_handler = None
    try:
        save_api_keys(payload.api_keys)
        write_default_language(payload.language)

        from config.globals import reload_api_keys
        from run import process_json
        from agent.common import logger as agent_logger

        reload_api_keys()
        log_handler = JobLogHandler(job)
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        agent_logger.logger.addHandler(log_handler)

        job.publish({"type": "status", "status": "running"})
        job.publish({"type": "log", "message": f"Model: {payload.model}"})
        job.publish({"type": "log", "message": f"Language: {payload.language}"})

        def on_progress(index: int, total: int, aggregate_model_stats: dict[str, Any], **_: Any) -> None:
            job.publish(
                {
                    "type": "progress",
                    "current": index,
                    "total": total,
                    "summary": {
                        "model": payload.model,
                        "language": payload.language,
                        "input_path": payload.input_path,
                        "output_path": payload.output_path,
                        "items_processed": index,
                        "items_total": total,
                        "model_stats": aggregate_model_stats,
                    },
                }
            )

        with redirect_stdout(stream), redirect_stderr(stream):
            _, summary = process_json(
                input_path=payload.input_path,
                output_path=payload.output_path,
                llm=payload.model,
                language=payload.language,
                progress_callback=on_progress,
            )
        summary["items_total"] = summary["items_processed"]
        job.publish({"type": "complete", "summary": summary})
    except Exception as exc:
        job.publish({"type": "error", "error": str(exc)})
        job.publish({"type": "log", "message": f"Error: {exc}"})
    finally:
        stream.flush()
        if log_handler is not None:
            try:
                from agent.common import logger as agent_logger

                agent_logger.logger.removeHandler(log_handler)
            except Exception:
                pass


def page_html() -> str:
    return """
<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>EASE Web UI</title>
<style>
:root{--bg:#090f1d;--panel:#182233;--panel2:#212c3e;--panel3:#0e1626;--line:#35435b;--line2:#2b364a;--text:#f3f7ff;--muted:#98a7bf;--accent:#7fb0ff;--accent2:#93f2c2;--warn:#ffb656;--danger:#ff7272;--shadow:0 24px 60px rgba(0,0,0,.35);--ui-font:"Helvetica Neue",Helvetica,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",Arial,sans-serif}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,rgba(127,176,255,.16),transparent 24%),radial-gradient(circle at top right,rgba(147,242,194,.08),transparent 24%),linear-gradient(180deg,#0b1120,#09101b);color:var(--text);font-family:var(--ui-font);font-weight:400;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}
.wrap{max-width:1680px;margin:0 auto;padding:28px 28px 34px}.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:20px}.title{font-size:34px;font-weight:700;letter-spacing:-.035em;margin:0;font-family:var(--ui-font)}.subtitle{margin:10px 0 0;color:var(--muted);max-width:760px;line-height:1.7;font-size:15px;font-weight:400}
.switch{display:inline-flex;gap:8px;padding:6px;background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:999px;backdrop-filter:blur(12px)}.switch button{border:0;background:transparent;color:var(--muted);padding:10px 16px;border-radius:999px;font-weight:700;cursor:pointer}.switch button.active{background:#253146;color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)}
.layout{display:grid;grid-template-columns:minmax(420px,1fr) minmax(520px,1fr);gap:20px}.panel{position:relative;background:linear-gradient(180deg,rgba(34,45,65,.92),rgba(26,35,51,.96));border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);overflow:hidden}
.panel::before{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.04),transparent 120px);pointer-events:none}.panel-head{display:flex;align-items:center;justify-content:space-between;padding:18px 18px 0 18px}.panel-badge{display:inline-flex;align-items:center;gap:8px;font-weight:600;padding:9px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(8,13,24,.22);font-family:var(--ui-font);letter-spacing:-.01em}.panel-desc{padding:10px 18px 0 18px;color:var(--muted);line-height:1.65;font-size:14px}
.panel-body{padding:18px}.config-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.row{display:flex;flex-direction:column;gap:7px}.row.full{grid-column:1/-1}label{font-size:13px;font-weight:600;color:#dce7f8;letter-spacing:-.01em}
input,select{width:100%;padding:13px 14px;border-radius:12px;border:1px solid var(--line2);background:rgba(13,21,36,.9);color:var(--text);outline:none;transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease;font-family:var(--ui-font);font-size:14px;font-weight:400;letter-spacing:-.01em}
input:focus,select:focus{border-color:#5f8fdb;box-shadow:0 0 0 4px rgba(95,143,219,.16)}input::placeholder{color:#73839c}
.hint{margin:0;color:var(--muted);font-size:12px;line-height:1.6;font-weight:400}.actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px}.actions button{border:0;border-radius:12px;padding:13px 18px;font-weight:600;cursor:pointer;font-family:var(--ui-font);letter-spacing:-.01em}
.primary{background:linear-gradient(135deg,#6f9cf6,#89b8ff);color:#0b1220}.secondary{background:rgba(255,255,255,.05);color:#edf4ff;border:1px solid var(--line)}.ghost{background:rgba(147,242,194,.08);color:#bcf5db;border:1px solid rgba(147,242,194,.18)}
.viewer{min-height:780px;display:flex;flex-direction:column}.split{display:grid;grid-template-columns:1.05fr 1fr;gap:18px;flex:1;align-items:start}.subpanel{min-height:560px;background:rgba(9,15,29,.36);border:1px solid var(--line);border-radius:14px;padding:16px;display:flex;flex-direction:column}
.subhead{display:inline-flex;align-items:center;gap:8px;width:max-content;padding:9px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.03);font-weight:600;margin-bottom:12px;letter-spacing:-.01em}.subdesc{color:var(--muted);font-size:13px;line-height:1.6;margin-bottom:12px;font-weight:400}
.stats{display:grid;grid-template-columns:repeat(2,minmax(140px,1fr));gap:12px}.stat{background:#10192a;border:1px solid var(--line);border-radius:14px;padding:14px}.k{font-size:12px;color:var(--muted);font-weight:400}.v{font-size:24px;font-weight:700;margin-top:8px;letter-spacing:-.03em}
.progline{display:flex;justify-content:space-between;align-items:center;margin:14px 0 8px;color:var(--muted);font-size:13px}.bar{height:10px;background:#0d1524;border-radius:999px;border:1px solid var(--line);overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,#7fb0ff,#93f2c2)}
.meta{display:grid;grid-template-columns:repeat(2,minmax(140px,1fr));gap:12px;margin-top:14px}.meta>div{padding:12px;border-radius:12px;border:1px solid var(--line);background:#10192a}.meta .val{margin-top:6px;font-weight:500;word-break:break-all;letter-spacing:-.01em}
.log-panel{min-height:auto}.log-resize{resize:vertical;overflow:auto;min-height:320px;height:470px;max-height:78vh;border-radius:14px}.log{height:100%;min-height:100%;background:#0b1220;border:1px solid var(--line);border-radius:14px;padding:16px;overflow:auto;white-space:pre-wrap;font:13px/1.65 Consolas,monospace;color:#d9e5ff}.drag-tip{margin:-2px 0 10px;color:#7f90aa;font-size:12px}
.disclaimer{margin-top:22px;padding:22px;border-radius:16px;border:1px solid rgba(255,182,86,.2);background:linear-gradient(180deg,rgba(255,182,86,.06),rgba(255,255,255,.02))}.disc-title{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700;color:#ffe1b4;letter-spacing:-.02em}.disc-text{margin-top:10px;color:#cfdae9;line-height:1.9;font-size:15px;font-weight:400}
.footer{margin-top:18px;text-align:right;color:#8091ab;font-size:14px}.error{color:var(--danger)!important;font-weight:700}.hidden{display:none!important}
@media (max-width:1220px){.layout{grid-template-columns:1fr}.split{grid-template-columns:1fr}.viewer{min-height:auto}}@media (max-width:760px){.wrap{padding:18px}.topbar{flex-direction:column}.config-grid,.meta,.stats{grid-template-columns:1fr}.actions button{width:100%}}
</style></head><body><div class='wrap'>
<div class='topbar'><div><h1 class='title' id='title'></h1><p class='subtitle' id='desc'></p></div><div class='switch'><button id='zhBtn'>中文</button><button id='enBtn'>English</button></div></div>
<div class='layout'>
<section class='panel'><div class='panel-head'><div class='panel-badge' id='cfgTitle'></div></div><div class='panel-desc' id='cfgDesc'></div><div class='panel-body'>
<div class='config-grid'>
<div class='row'><label id='modelLabel' for='model'></label><select id='model'></select></div>
<div class='row'><label id='langLabel' for='langState'></label><input id='langState' disabled></div>
<div class='row full'><label id='inputLabel' for='inputPath'></label><input id='inputPath'></div>
<div class='row full'><label id='outputLabel' for='outputPath'></label><input id='outputPath'></div>
<div class='row full'><label id='openaiKeyLabel' for='openaiKey'></label><input id='openaiKey' type='password'></div>
<div class='row full hidden' id='openaiBaseRow'><label id='openaiBaseLabel' for='openaiBaseUrl'></label><input id='openaiBaseUrl' placeholder='https://api.openai.com/v1'><p class='hint' id='openaiBaseHint'></p></div>
<div class='row full'><label id='deepseekKeyLabel' for='deepseekKey'></label><input id='deepseekKey' type='password'></div>
<div class='row'><label id='serperKeyLabel' for='serperKey'></label><input id='serperKey' type='password'></div>
<div class='row'><label id='hfKeyLabel' for='hfKey'></label><input id='hfKey' type='password'></div>
</div>
<div class='actions'><button class='secondary' id='saveBtn'></button><button class='primary' id='runBtn'></button><button class='ghost' id='syncBtn'></button></div>
<p class='hint' style='margin-top:14px' id='statusLine'></p></div></section>
<section class='panel viewer'><div class='panel-head'><div class='panel-badge' id='viewerTitle'></div></div><div class='panel-desc' id='viewerDesc'></div><div class='panel-body' style='padding-top:14px;flex:1'>
<div class='split'>
<div class='subpanel'><div class='subhead' id='monitorTitle'></div><div class='subdesc' id='monitorDesc'></div>
<div class='stats'><div class='stat'><div class='k' id='statusKey'></div><div class='v' id='statusVal'>Idle</div></div><div class='stat'><div class='k' id='costKey'></div><div class='v' id='costVal'>$0.000000</div></div><div class='stat'><div class='k' id='inKey'></div><div class='v' id='inVal'>0</div></div><div class='stat'><div class='k' id='outKey'></div><div class='v' id='outVal'>0</div></div></div>
<div class='progline'><span id='progressLabel'></span><span id='progressText'>0 / 0</span></div><div class='bar'><div class='fill' id='fill'></div></div>
<div class='meta'><div><div class='k' id='curModelKey'></div><div class='val' id='curModel'>-</div></div><div><div class='k' id='curLangKey'></div><div class='val' id='curLang'>-</div></div><div><div class='k' id='inFileKey'></div><div class='val' id='inFile'>-</div></div><div><div class='k' id='outFileKey'></div><div class='val' id='outFile'>-</div></div></div>
<p class='hint' style='margin-top:12px' id='errorLine'></p></div>
<div class='subpanel log-panel'><div class='subhead' id='logTitle'></div><div class='subdesc' id='logDesc'></div><div class='drag-tip' id='dragTip'></div><div class='log-resize'><div class='log' id='logBox'></div></div></div>
</div>
</div></section></div>
<section class='disclaimer'><div class='disc-title' id='discTitle'></div><div class='disc-text' id='discText'></div></section><div class='footer' id='footerText'></div>
<script>
const text={zh:{title:"EASE Agent 控制台",desc:"为事实核验流程提供接近产品化的深色控制台体验，支持配置模型、API、语言与实时观察费用、日志、进度。",cfgTitle:"Claims",cfgDesc:"在这里准备运行参数。若选择 GPT 系列模型，可以填写兼容 OpenAI 接口的第三方 Base URL；留空时默认走 OpenAI 官方 API 地址 https://api.openai.com/v1 。",viewerTitle:"Verification Result",viewerDesc:"右侧区域用于实时承载任务状态、费用变化和日志流。布局和观感参考你给的截图做了统一的深色面板化处理。",modelLabel:"模型",langLabel:"当前语言",inputLabel:"输入 JSON 路径",outputLabel:"输出 JSON 路径",openaiKeyLabel:"OpenAI API Key",openaiBaseLabel:"OpenAI 兼容 Base URL",openaiBaseHint:"仅在 GPT 系列模型下生效。示例：https://api.openai.com/v1 或第三方兼容地址。",deepseekKeyLabel:"DeepSeek API Key",serperKeyLabel:"Serper API Key",hfKeyLabel:"HuggingFace Token",saveBtn:"保存配置",runBtn:"启动任务",syncBtn:"同步语言到 run.py",monitorTitle:"Live Monitor",monitorDesc:"成本按 available_models.csv 中的单价和真实 token 使用量累计。",statusKey:"状态",costKey:"累计费用",inKey:"输入 Tokens",outKey:"输出 Tokens",progressLabel:"处理进度",curModelKey:"当前模型",curLangKey:"当前语言",inFileKey:"输入文件",outFileKey:"输出文件",logTitle:"运行日志",logDesc:"这里实时显示 run.py 输出、Agent 日志和异常信息。",dragTip:"可拖动右下角拉伸日志高度",discTitle:"⚠ Disclaimer",discText:"AI 核验的效率会受到检索质量、来源可信度评估以及大模型语言理解能力的影响。系统会尽量收集可用证据并给出辅助判断，但仍可能出现误差，请结合你的实际场景进行最终决策。",footerText:"Use via API · 使用 FastAPI 构建",idle:"空闲",queued:"排队中",running:"运行中",completed:"已完成",failed:"失败",saved:"配置已保存",saving:"正在保存配置...",starting:"任务已启动，正在连接实时日志流",synced:"语言已同步到 run.py",languageZh:"中文",languageEn:"英文"},en:{title:"EASE Agent Console",desc:"A product-style dark control room for fact-checking runs, with model settings, API configuration, language switching, and real-time cost, logs, and progress.",cfgTitle:"Claims",cfgDesc:"Configure your run here. When a GPT family model is selected, you can provide any OpenAI-compatible third-party Base URL; if left blank, the app falls back to the official OpenAI API endpoint: https://api.openai.com/v1 .",viewerTitle:"Verification Result",viewerDesc:"The right side streams live job state, cost updates, and runtime logs in a darker interface inspired by your reference image.",modelLabel:"Model",langLabel:"Current Language",inputLabel:"Input JSON Path",outputLabel:"Output JSON Path",openaiKeyLabel:"OpenAI API Key",openaiBaseLabel:"OpenAI-Compatible Base URL",openaiBaseHint:"Only used for GPT family models. Example: https://api.openai.com/v1 or any compatible third-party endpoint.",deepseekKeyLabel:"DeepSeek API Key",serperKeyLabel:"Serper API Key",hfKeyLabel:"HuggingFace Token",saveBtn:"Save Config",runBtn:"Start Run",syncBtn:"Sync Language to run.py",monitorTitle:"Live Monitor",monitorDesc:"Cost is accumulated from available_models.csv pricing and real token usage.",statusKey:"Status",costKey:"Total Cost",inKey:"Input Tokens",outKey:"Output Tokens",progressLabel:"Progress",curModelKey:"Current Model",curLangKey:"Language",inFileKey:"Input File",outFileKey:"Output File",logTitle:"Run Logs",logDesc:"This area streams run.py output, Agent logs, and runtime errors.",dragTip:"Drag the bottom-right corner to resize the log area",discTitle:"⚠ Disclaimer",discText:"AI-assisted verification quality can vary with search quality, source credibility assessment, and LLM language understanding. The system is designed to collect relevant evidence and provide useful recommendations, but errors are still possible, so final judgment should remain yours.",footerText:"Use via API · Built with FastAPI",idle:"Idle",queued:"Queued",running:"Running",completed:"Completed",failed:"Failed",saved:"Configuration saved",saving:"Saving configuration...",starting:"Run started. Connecting live stream",synced:"Language synced to run.py",languageZh:"Chinese",languageEn:"English"}};
let lang='zh',source=null,last=null,models=[];
const $=id=>document.getElementById(id);
function t(k){return text[lang][k]||k}
function isGptModel(v){return String(v||'').startsWith('gpt_')}
function toggleOpenAIBase(){const show=isGptModel($('model').value);$('openaiBaseRow').classList.toggle('hidden',!show)}
function applyText(){['title','desc','cfgTitle','cfgDesc','viewerTitle','viewerDesc','modelLabel','langLabel','inputLabel','outputLabel','openaiKeyLabel','openaiBaseLabel','openaiBaseHint','deepseekKeyLabel','serperKeyLabel','hfKeyLabel','saveBtn','runBtn','syncBtn','monitorTitle','monitorDesc','statusKey','costKey','inKey','outKey','progressLabel','curModelKey','curLangKey','inFileKey','outFileKey','logTitle','logDesc','dragTip','discTitle','discText','footerText'].forEach(id=>$(id).textContent=t(id));$('zhBtn').textContent=t('languageZh');$('enBtn').textContent=t('languageEn');$('zhBtn').classList.toggle('active',lang==='zh');$('enBtn').classList.toggle('active',lang==='en');$('langState').value=lang;toggleOpenAIBase();render(last)}
function statusText(s){return t(s||'idle')}
function money(v){return '$'+Number(v||0).toFixed(6)}
function num(v){return Number(v||0).toLocaleString()}
function payload(){return{api_keys:{openai_api_key:$('openaiKey').value.trim(),openai_base_url:$('openaiBaseUrl').value.trim(),deepseek_api_key:$('deepseekKey').value.trim(),serper_api_key:$('serperKey').value.trim(),huggingface_user_access_token:$('hfKey').value.trim()},language:lang}}
function render(s){last=s;const sum=s?.summary||{},m=sum.model_stats||{},cur=s?.progress_current||0,total=s?.progress_total||sum.items_total||0,pct=total?Math.min(100,cur/total*100):0;$('statusVal').textContent=statusText(s?.status);$('costVal').textContent=money(m['Total cost']);$('inVal').textContent=num(m['Input tokens']);$('outVal').textContent=num(m['Output tokens']);$('progressText').textContent=`${cur} / ${total}`;$('fill').style.width=pct+'%';$('curModel').textContent=sum.model||'-';$('curLang').textContent=sum.language||lang;$('inFile').textContent=sum.input_path||$('inputPath').value||'-';$('outFile').textContent=sum.output_path||$('outputPath').value||'-';$('errorLine').textContent=s?.error||'';$('errorLine').className=s?.error?'hint error':'hint'}
function addLog(msg){if(!msg)return;const box=$('logBox');box.textContent+=(box.textContent?'\\n':'')+msg;box.scrollTop=box.scrollHeight}
async function load(){const r=await fetch('/api/config');const d=await r.json();models=d.models||[];lang=d.language||'zh';$('openaiKey').value=d.api_keys.openai_api_key||'';$('openaiBaseUrl').value=d.api_keys.openai_base_url||'';$('deepseekKey').value=d.api_keys.deepseek_api_key||'';$('serperKey').value=d.api_keys.serper_api_key||'';$('hfKey').value=d.api_keys.huggingface_user_access_token||'';$('inputPath').value=d.defaults.input_path;$('outputPath').value=d.defaults.output_path;$('model').innerHTML='';models.forEach(m=>{const o=document.createElement('option');o.value=m.shorthand;o.textContent=`${m.shorthand} · ${m.name} · $${m.input_cost_per_m}/$${m.output_cost_per_m}`;if(m.shorthand===d.defaults.model)o.selected=true;$('model').appendChild(o)});$('langState').value=lang;applyText();render(null)}
async function save(){ $('statusLine').textContent=t('saving'); const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())}); if(!r.ok){const e=await r.json();throw new Error(e.detail||'save failed')}$('statusLine').textContent=t('saved')}
async function syncLanguage(){await save();$('statusLine').textContent=t('synced')}
function closeStream(){if(source){source.close();source=null}}
function connect(jobId){closeStream();source=new EventSource(`/api/jobs/${jobId}/stream`);source.onmessage=e=>{const p=JSON.parse(e.data);if(p.type==='heartbeat')return;if(p.type==='snapshot'){$('logBox').textContent=(p.data.logs||[]).join('\\n');render(p.data)}if(p.type==='log')addLog(p.message);if(p.type==='status')render({...last,status:p.status});if(p.type==='progress')render({...last,status:'running',progress_current:p.current,progress_total:p.total,summary:p.summary});if(p.type==='complete'){render({...last,status:'completed',summary:p.summary});closeStream()}if(p.type==='error'){render({...last,status:'failed',error:p.error});closeStream()}}}
async function runTask(){closeStream();$('logBox').textContent='';render({status:'queued',summary:{model:$('model').value,language:lang,input_path:$('inputPath').value.trim(),output_path:$('outputPath').value.trim(),model_stats:{}}});const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload(),input_path:$('inputPath').value.trim(),output_path:$('outputPath').value.trim(),model:$('model').value})});if(!r.ok){const e=await r.json();throw new Error(e.detail||'run failed')}const d=await r.json();$('statusLine').textContent=t('starting');connect(d.job_id)}
$('model').addEventListener('change',toggleOpenAIBase);$('zhBtn').onclick=()=>{lang='zh';applyText()};$('enBtn').onclick=()=>{lang='en';applyText()};$('saveBtn').onclick=()=>save().catch(e=>$('statusLine').textContent=String(e.message));$('syncBtn').onclick=()=>syncLanguage().catch(e=>$('statusLine').textContent=String(e.message));$('runBtn').onclick=()=>runTask().catch(e=>$('statusLine').textContent=String(e.message));load().catch(e=>$('statusLine').textContent=String(e.message));
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return page_html()


@app.get("/api/config")
def get_config() -> JSONResponse:
    return JSONResponse(
        {
            "api_keys": load_api_keys(),
            "language": read_default_language(),
            "models": load_models(),
            "defaults": {
                "input_path": str(DEFAULT_INPUT_PATH),
                "output_path": str(DEFAULT_OUTPUT_PATH),
                "model": "deepseek_chat",
            },
        }
    )


@app.post("/api/config")
def update_config(payload: ConfigPayload) -> JSONResponse:
    payload.language = validate_language(payload.language)
    save_api_keys(payload.api_keys)
    write_default_language(payload.language)
    return JSONResponse({"ok": True})


@app.post("/api/run")
def start_run(payload: RunPayload) -> JSONResponse:
    payload.language = validate_language(payload.language)
    if not Path(payload.input_path).exists():
        raise HTTPException(status_code=400, detail=f"Input file not found: {payload.input_path}")
    job = JobState(job_id=str(uuid.uuid4()))
    jobs[job.job_id] = job
    threading.Thread(target=run_job, args=(job, payload), daemon=True).start()
    return JSONResponse({"job_id": job.job_id})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.snapshot())


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        subscriber = job.add_subscriber()
        try:
            yield f"data: {json.dumps({'type': 'snapshot', 'data': job.snapshot()}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield "data: {\"type\":\"heartbeat\"}\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in {"complete", "error"}:
                    break
        finally:
            job.remove_subscriber(subscriber)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
