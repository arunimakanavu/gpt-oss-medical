from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import torch, re

# ---------------------------
# Load base + adapter
# ---------------------------
base_model = "openai/gpt-oss-20b"
adapter = "arunimas1107/gpt-oss-medical"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # use bf16 to save memory
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# ---------------------------
# Load MedMCQA test split
# ---------------------------
dataset = load_dataset("medmcqa", split="test")

# Map index → option letter
idx2opt = {0: "A", 1: "B", 2: "C", 3: "D"}

# ---------------------------
# Evaluation Loop
# ---------------------------
correct, total = 0, 0

for sample in dataset:  
    question = sample["question"]
    options = [sample["opa"], sample["opb"], sample["opc"], sample["opd"]]

    # Gold label
    try:
        gold_idx = int(sample["cop"])
        gold_answer = idx2opt.get(gold_idx, None)
    except:
        gold_answer = None
    if gold_answer is None:
        continue  # skip if label is missing

    # Prompt
    prompt = f"""Question: {question}
Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer with only the correct option letter (A, B, C, or D)."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract first valid option letter
    match = re.search(r"\b[A-D]\b", answer)
    pred = match.group(0) if match else None

    # Update stats
    if pred == gold_answer:
        correct += 1
    total += 1

# ---------------------------
# Final Accuracy
# ---------------------------
accuracy = correct / total if total > 0 else 0.0
print(f"Evaluated on {total} samples")
print(f"Accuracy: {accuracy:.2%}")
