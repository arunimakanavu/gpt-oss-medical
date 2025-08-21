from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./medical_gpt_oss_20b_final")
tokenizer = AutoTokenizer.from_pretrained("./medical_gpt_oss_20b_final", trust_remote_code=True)

prompt = "Patient shows persistent cough, chest tightness, and low-grade fever. Suggest possible diagnoses."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=1024)
    print(tokenizer.decode(output[0], skip_special_tokens=True))