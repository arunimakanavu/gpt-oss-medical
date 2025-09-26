from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Load base + adapter
base_model = "openai/gpt-oss-20b"
adapter = "arunimas1107/gpt-oss-medical"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load MedMCQA test split
dataset = load_dataset("medmcqa", split="test")

# Mapping index to letter
idx2opt = {0: "A", 1: "B", 2: "C", 3: "D"}

for sample in dataset.select(range(5)):  
    question = sample["question"]
    options = [sample["opa"], sample["opb"], sample["opc"], sample["opd"]]
    gold_idx = sample["cop"]
    gold_answer = idx2opt[gold_idx] if gold_idx in idx2opt else "Unknown"

    # Prompt construction
    prompt = f"Question: {question}\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the correct option letter."

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Q: {question}\nModel: {answer}\nGold: {gold_answer}\n")