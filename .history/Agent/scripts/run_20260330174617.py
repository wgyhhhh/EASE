import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

os.environ["LANGUAGE"] = "zh"  # Set language to Chinese, you can change it to "en" for English

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from agent.fact_checker import Agent  # noqa: E402

DEFAULT_MODEL = "deepseek_chat"
DEFAULT_INPUT_PATH = project_root / "data" / "input.json"
DEFAULT_OUTPUT_PATH = project_root / "data" / "output.json"


def set_language(language: str) -> str:
    normalized = (language or "zh").lower()
    if normalized not in {"zh", "en"}:
        raise ValueError("Language must be 'zh' or 'en'.")
    os.environ["LANGUAGE"] = normalized
    return normalized


def process_json(
    input_path: str | os.PathLike = DEFAULT_INPUT_PATH,
    output_path: str | os.PathLike = DEFAULT_OUTPUT_PATH,
    llm: str = DEFAULT_MODEL,
    language: str | None = None,
    progress_callback=None,
):
    if language is not None:
        set_language(language)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fact_checker = Agent(llm=llm)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    aggregate_model_stats = {
        "Calls": 0,
        "Input tokens": 0,
        "Output tokens": 0,
        "Input tokens cost": 0.0,
        "Output tokens cost": 0.0,
        "Total cost": 0.0,
    }

    total_items = len(data)
    for index, item in enumerate(tqdm(data, desc="Processing", unit="item"), start=1):
        claim_text = item.get("content", "")
        result, meta = fact_checker.verify_claim([claim_text])

        item["evidence"] = result.justification if result.justification else ""
        item["evidence_pred"] = result.verdict.value if result.verdict else ""
        item["reasoning"] = result.reasoning_justification if result.reasoning_justification else ""
        item["reasoning_pred"] = result.reason_verdict.value if result.reason_verdict else ""
        item["sentiment"] = result.sentiment_justification if result.sentiment_justification else ""
        item["sentiment_pred"] = result.sentiment_verdict.value if result.sentiment_verdict else ""

        if "label" in item:
            label = item["label"]
            original = label if isinstance(label, str) else getattr(label, "name", str(label))
            item["evidence_acc"] = 1 if item["evidence_pred"] == original else 0
            item["reasoning_acc"] = 1 if item["reasoning_pred"] == original else 0
            item["sentiment_acc"] = 1 if item["sentiment_pred"] == original else 0
        else:
            item["evidence_acc"] = 0
            item["reasoning_acc"] = 0
            item["sentiment_acc"] = 0

        model_stats = meta.get("Statistics", {}).get("Model", {})
        for key in aggregate_model_stats:
            aggregate_model_stats[key] += model_stats.get(key, 0)

        if progress_callback is not None:
            progress_callback(
                index=index,
                total=total_items,
                item=item,
                meta=meta,
                aggregate_model_stats=dict(aggregate_model_stats),
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    summary = {
        "model": llm,
        "language": os.environ.get("LANGUAGE", "zh"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "items_processed": len(data),
        "model_stats": aggregate_model_stats,
    }
    return data, summary


def main():
    _, summary = process_json()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
