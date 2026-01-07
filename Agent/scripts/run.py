import os, sys, json
os.environ["LANGUAGE"] = "zh" # Set language to Chinese, you can change it to "en" for English

from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from agent.fact_checker import Agent


def process_json(input_path, output_path):
    fact_checker = Agent(llm="gpt_4o_mini")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data, desc="Processing", unit="item"):
        claim_text = item.get("content", "")
        

        result, _ = fact_checker.verify_claim([claim_text])

        item["evidence"] = result.justification if result.justification else ""

        if result.verdict:
            item["evidence_pred"] = result.verdict.value
        else:
            item["evidence_pred"] = ""

        item["reasoning"] = result.reasoning_justification if result.reasoning_justification else ""

        if result.reason_verdict:
            item["reasoning_pred"] = result.reason_verdict.value
        else:
            item["reasoning_pred"] = ""

        item["sentiment"] = result.sentiment_justification if result.sentiment_justification else ""

        if result.sentiment_verdict:
            item["sentiment_pred"] = result.sentiment_verdict.value
        else:
            item["sentiment_pred"] = ""

        if "label" in item:
            label = item["label"]
            original = label if isinstance(label, str) else getattr(label, 'name', str(label))
            predicted_evidence = item["evidence_pred"]
            item["evidence_acc"] = 1 if predicted_evidence == original else 0
        else:
            item["evidence_acc"] = 0

        if "label" in item:
            label = item["label"]
            original = label if isinstance(label, str) else getattr(label, 'name', str(label))
            predicted_reasoning = item["reasoning_pred"]
            item["reasoning_acc"] = 1 if predicted_reasoning == original else 0
        else:
            item["reasoning_acc"] = 0

        if "label" in item:
            label = item["label"]
            original = label if isinstance(label, str) else getattr(label, 'name', str(label))
            predicted_sentiment = item["sentiment_pred"]
            item["sentiment_acc"] = 1 if predicted_sentiment == original else 0
        else:
            item["sentiment_acc"] = 0    
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

process_json("/home/test3/test3/test3/wgy/DEFAME-main/data/input.json", "/home/test3/test3/test3/wgy/DEFAME-main/data/inpu1t.json")