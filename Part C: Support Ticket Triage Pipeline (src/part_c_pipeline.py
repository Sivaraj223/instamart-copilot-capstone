import os
import re
import json
import time
import random
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT_PROMPT = """CLASSIFY the customer support ticket into exactly one of: billing, delivery, technical, fraud.
Reply in strict JSON format: {"category": "<category>", "reason": "<short explanation>"}"""

FEW_SHOT_PROMPT = """CLASSIFY the customer support ticket into exactly one of: billing, delivery, technical, fraud.
Reply in strict JSON format: {"category": "<category>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: Unauthorized money was deducted from my account.
{"category": "fraud", "reason": "unauthorized charge"}

Ticket: Rider has not arrived yet.
{"category": "delivery", "reason": "order delayed"}"""

def redact_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    text = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text = re.sub(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text

def resilient_call(messages, ticket_id):
    for n in range(3):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            # Log single line as required
            print(f"Ticket {ticket_id} retry {n} waiting {wait:.4f}s")
            time.sleep(wait)
    raise RuntimeError(f"Permanent failure on {ticket_id}")

def validate_response(resp_content):
    try:
        data = json.loads(resp_content)
        if isinstance(data, dict) and data.get("category") in ALLOWED_CATEGORIES:
            return "classified_json", data["category"]
    except Exception:
        pass
    
    # Word regex extraction
    words = re.findall(r"[a-z]+", resp_content.lower())
    matched = [c for c in ALLOWED_CATEGORIES if c in words]
    if len(matched) == 1:
        return "classified_keyword", matched[0]
        
    return "flagged_invalid_output", None

def run_triage(is_few_shot=True, token_budget=100000):
    mock_llm.reset()
    tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    segments_df = pd.read_csv("outputs/customer_segments.csv")
    prem_map = dict(zip(segments_df['customer_id'], segments_df['is_premium']))
    seg_name_map = dict(zip(segments_df['customer_id'], segments_df['segment_name']))
    
    sys_prompt = FEW_SHOT_PROMPT if is_few_shot else ZERO_SHOT_PROMPT
    running_tokens = 0
    results = []
    
    for _, row in tickets_df.iterrows():
        tid = row['ticket_id']
        cid = row['customer_id']
        txt = row['text']
        
        # 1. Input Guardrails
        if not txt or txt.strip() == "":
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "segment_name": seg_name_map.get(cid, ""),
                "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        if mock_llm.est_tokens(txt) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "segment_name": seg_name_map.get(cid, ""),
                "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        # 2. Token Budget check
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "segment_name": seg_name_map.get(cid, ""),
                "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        # Redact PII
        redacted = redact_text(txt)
        assert not re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', redacted)
        assert not re.search(r'\b[6-9]\d{9}\b', redacted)
        assert not re.search(r'ORD-\d{6}', redacted)
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": redacted}
        ]
        
        resp = resilient_call(messages, tid)
        call_tokens = resp.usage.total_tokens
        running_tokens += call_tokens
        
        status, category = validate_response(resp.choices[0].message.content)
        
        # Routing Logic
        is_premium = prem_map.get(cid, False)
        seg_name = seg_name_map.get(cid, "")
        
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
            
        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": category if category else "",
            "segment_name": seg_name, "priority": priority,
            "route": route, "tokens_used": call_tokens
        })
        
    return pd.DataFrame(results)

def run_part_c():
    os.makedirs("outputs", exist_ok=True)
    # Budget Demo run
    budget_df = run_triage(is_few_shot=False, token_budget=6000)
    budget_df.to_csv("outputs/triage_budget_demo.csv", index=False)
    
    # Final Few-shot production run
    final_df = run_triage(is_few_shot=True, token_budget=100000)
    final_df.to_csv("outputs/triage_results.csv", index=False)

if __name__ == "__main__":
    run_part_c()
