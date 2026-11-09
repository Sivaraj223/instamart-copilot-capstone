# Instamart Customer Intelligence & Support Copilot

An end-to-end data, machine learning, and LLM-driven copilot built to resolve raw order export discrepancies, predict customer churn/lapse, triage high-volume support tickets under strict PII guardrails, and automate policy question-answering and refund escalation.

---

## Reproduce

Execute the following commands in sequence on a clean terminal:

```bash
# 1. Environment creation and activation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate raw synthetic data
python generate_data.py

# 4. Part A: Cleaning, validation and database storage
python src/part_a_pipeline.py
python src/part_a_validate.py

# 5. Part B: RFM segmentation and lapse model training
python src/part_b_pipeline.py

# 6. Part C: Support ticket triage pipeline and guardrails
python src/part_c_pipeline.py

# 7. Part D: Policy RAG evaluation and ReAct refund agent
python src/part_d_pipeline.py
