---

base\_model: openai/gpt-oss-20b 
library\_name: peft 
pipeline\_tag: text-generation 
tags:
- base\_model\:adapter\:openai/gpt-oss-20b
- lora
- transformers

---

# Model Card for Medical GPT-OSS-20B LoRA Adapter

## Model Details

### Model Description

This model is a LoRA adapter fine-tuned on **openai/gpt-oss-20b** using the PEFT library. It is optimized for **medical domain tasks** such as question answering, summarization, and knowledge retrieval in healthcare contexts. The adapter modifies the base model with efficient fine-tuning techniques while retaining the general-purpose reasoning capabilities of the underlying 20B parameter model.

- **Developed by:** Arunima Surendran
- **Funded by:** E2E Cloud
- **Shared by:** Arunima Surendran
- **Model type:** Large Language Model with LoRA adapter (20B base)
- **Language(s):** English
- **License:** Apache 2.0
- **Finetuned from model:** openai/gpt-oss-20b
- **Finetuned using:** NVIDIA 2xH200 for 12+ hours on E2E Cloud TIR Instance

### Model Sources

- **Repository:** [https://github.com/arunimakanavu/gpt-oss-medical](https://github.com/arunimakanavu/gpt-oss-medical)
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Requirements

```text
torch>=2.0.0
transformers @ git+https://github.com/huggingface/transformers.git
datasets>=2.12.0
peft>=0.10.0
accelerate>=0.22.0
bitsandbytes>=0.41.0
sentencepiece>=0.1.99
```

## Uses

### Direct Use

- Medical Q&A
- Clinical text summarization
- Educational content generation in healthcare

### Downstream Use

- Integrating into RAG pipelines with domain-specific medical knowledge bases
- Deployment in medical chatbots (for informational purposes only)

### Out-of-Scope Use

- Direct clinical decision-making or diagnostic tools without human oversight
- High-stakes medical applications without proper validation

## Bias, Risks, and Limitations

The model may generate:

- Inaccurate or hallucinated medical information
- Biased outputs due to limitations in training data
- Text not suitable for unsupervised clinical decision-making

### Recommendations

Users should:

- Treat outputs as **assistive**, not authoritative
- Always cross-verify with trusted medical sources
- Avoid using the model for patient-facing diagnosis without professional review

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel

base_model = "openai/gpt-oss-20b"
adapter_path = "./medical_gpt_oss_20b_final"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(base_model)
```

## Training Details

### Training Data

- Domain-specific medical text corpus
- Filtered for quality and relevance

### Training Procedure

- Fine-tuned using **LoRA** on top of GPT-OSS-20B
- Mixed precision training (`bf16`)

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **Compute:** NVIDIA 2xH200 GPUs
- **Training time:** 12+ hours

## Evaluation

### Testing Data, Factors & Metrics

- Domain: medical Q&A and summarization
- Metrics: Perplexity, BLEU/ROUGE for summarization, accuracy for Q&A

### Results

- [Add evaluation metrics and benchmarks]

## Environmental Impact

- **Hardware Type:** NVIDIA 2xH200 (E2E TIR platform)
- **Hours used:** 12+
- **Cloud Provider:** E2E Networks
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications

### Model Architecture and Objective

- Base: GPT-OSS-20B (20 billion parameters)
- Adapter: LoRA (low-rank fine-tuning)

### Compute Infrastructure

- **Hardware:** NVIDIA 2xH200 (E2E TIR)
- **Software:** PyTorch, Transformers, PEFT

## Citation

**BibTeX:**

```bibtex
@misc{gptoss20b-medical,
  title = {Medical GPT-OSS-20B LoRA Adapter},
  author = {Arunima Surendran},
  year = {2025},
  url = {https://github.com/arunimakanavu/gpt-oss-medical}
}
```

## Model Card Authors

- Arunima Surendran

## Model Card Contact

- [Your Contact Information]

### Framework versions

- PEFT 0.17.0

