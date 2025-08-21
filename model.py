from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True  # very important for gpt_oss
)

tokenizer = AutoTokenizer.from_pretrained(
    "./medical_gpt_oss_20b_final",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./medical_gpt_oss_20b_final")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True  # very important for gpt_oss
)

tokenizer = AutoTokenizer.from_pretrained(
    "./medical_gpt_oss_20b_final",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./medical_gpt_oss_20b_final")
