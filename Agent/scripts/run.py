import os, sys, json
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent.fact_checker import FactChecker

def process_json(input_path, output_path):
    fact_checker = FactChecker(llm="gpt_4o")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data, desc="Processing", unit="item"):
        claim_text = item.get("content", "")
        

        result, _ = fact_checker.verify_claim([claim_text])

        item["evidence"] = result.justification if result.justification else ""

        if result.verdict:
            item["evidence_pred"] = result.verdict.name
        else:
            item["evidence_pred"] = ""

        if "label" in item:
            label = item["label"]
            original = label if isinstance(label, str) else getattr(label, 'name', str(label))
            predicted = item["evidence_pred"]
            item["evidence_acc"] = 1 if predicted == original else 0
        else:
            item["evidence_acc"] = 0
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

process_json("", "")