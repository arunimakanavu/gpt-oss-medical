# Model Evaluation Report

## Requirements

To reproduce this notebook, install the following dependencies:

* **Python**: 3.10+
* **Transformers**: `4.56.0.dev0`
* **PEFT**: `>=0.11.1`
* **Accelerate**: `>=0.31.0`
* **Torch**: `>=2.1.0` (with CUDA)
* **Datasets**: `>=2.20.0`
* **Evaluate**: `>=0.4.2`
* **Jupyter Notebook** (optional, for interactive execution)

**Resources used:**

* **NVIDIA H200 (single core)** GPU with CUDA acceleration.


## Step-by-Step Explanation

### 1. Load Dataset

```python
dataset = load_dataset("lavita/medical-qa-datasets", name="all-processed", split="train[:500]")
```

* Loads the **medical QA dataset** (`lavita/medical-qa-datasets`).
* Only a subset of 500 samples is used for quicker evaluation.


### 2. Load Model + Tokenizer

```python
base_model = "openai/gpt-oss-20b"
adapter = "arunimas1107/gpt-oss-medical"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter)
```

* **Base model:** [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
* **Adapter:** [`arunimas1107/gpt-oss-medical`](https://huggingface.co/arunimas1107/gpt-oss-medical) (fine-tuned on medical reasoning).
* Loads in **bfloat16** precision for GPU efficiency.

A Hugging Face `pipeline` is then created for text generation.


### 3. Metrics

```python
bertscore = evaluate.load("bertscore")
accuracy_metric = evaluate.load("accuracy")
```

* **BERTScore** → semantic similarity between prediction & reference.
* **Accuracy** → for multiple-choice questions (MCQs).


### 4. Utility Functions

* **`detect_mcq`** → detects if a sample is an MCQ by looking for options (A, B, C, D).
* **`make_prompt`** → formats input differently for MCQ vs free-text QA.
* **`evaluate_response`** →

  * For MCQ → checks if predicted letter matches gold answer.
  * For free-text → computes BERTScore.



### 5. Main Evaluation Loop

```python
for i, sample in enumerate(dataset):
    prompt = make_prompt(sample)
    output = pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
    prediction = output[len(prompt):].strip()
    metrics = evaluate_response(sample, prediction)
```

* Iterates over 500 test samples.
* Generates predictions sequentially on GPU.
* Collects metrics (`MCQ accuracy` and `BERTScore`).

⚠️ **Logs warning:** “You seem to be using the pipelines sequentially on GPU” → batching could improve speed.



### 6. Summarize Results

At the end:

```python
MCQ Accuracy: 11.11%
Avg BERTScore (F1): 0.834
```



## Output & Observations

### Runtime Logs

* Model shards loaded (`openai/gpt-oss-20b`).
* Device: `cuda:0`.
* Some warnings about **RobertaModel pooler weights** — irrelevant (noise from an internal dependency, not your actual model).
* Generation flag `temperature` ignored (pipeline uses deterministic generation with `do_sample=False`).

### Predictions (examples)

* `[0]` “Based on the information you provided, it sounds like the swelling in your client’s feet…”
* `[140]` `"HYPON: A Novel Hyperbolic Space Approach for Enhancing Biomedical Ontology Embedding"`
* `[400]` “Holt-Oram Syndrome is a rare genetic disorder that primarily affects the development of the upper limb…”

### Metrics

* **MCQ Accuracy = 11.11%**

  * Indicates weak performance on strict multiple-choice tasks.
* **BERTScore (F1) = 0.834**

  * Strong semantic overlap with references, even when exact match is low.
* **Empty reference warning** → some dataset samples lacked ground-truth answers, lowering reliability of metrics.



## Resources

* **Hardware**: NVIDIA H200 single-core GPU.
* **Efficiency Note**: Sequential pipeline calls → better batching or `model.generate()` batched would improve runtime.



## References

* **Dataset**: [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets)
* **Base Model**: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
* **Fine-Tuned Adapter**: [arunimas1107/gpt-oss-medical](https://huggingface.co/arunimas1107/gpt-oss-medical)
* **Transformers**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
* **PEFT**: [PEFT Library](https://github.com/huggingface/peft)
