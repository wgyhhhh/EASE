import os, sys, json
from tqdm import tqdm
from enum import Enum
from agent.fact_checker import FactChecker
from ezmm import Image

def process_json(input_path, output_path):
    fact_checker = FactChecker(llm="gpt_4o")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data, desc="Processing", unit="item"):
        claim_text = item.get("content", "")
        
        item["claim"] = [claim_text]
        result, _ = fact_checker.verify_claim(item["claim"])

        item["evidence"] = result.justification if result.justification else ""

        if hasattr(result.verdict, 'value'):
            item["evidence_pred"] = result.verdict.value  
        else:
            item["evidence_pred"] = str(result.verdict)

        if "label" in item:
            if hasattr(item["label"], 'value'):
                original_label = item["label"].value
            else:
                original_label = str(item["label"])
            
            if hasattr(result.verdict, 'value'):
                predicted_label = result.verdict.value
            else:
                predicted_label = str(result.verdict)
            
            item["evidence_acc"] = 1 if predicted_label == original_label else 0
        else:
            item["evidence_acc"] = 0  
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

process_json("input.json", "output.json")