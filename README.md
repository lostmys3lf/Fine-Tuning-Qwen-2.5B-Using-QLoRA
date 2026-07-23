# Fine-Tuning Qwen2.5-0.5B Using QLoRA

A hands-on notebook for fine-tuning a small language model (Qwen2.5-0.5B-Instruct) using **QLoRA (Quantized Low-Rank Adaptation)**. The full pipeline — loading the model in 4-bit, attaching a LoRA adapter, training, and before/after evaluation — is designed to run end-to-end on **Google Colab with a T4 GPU (free tier)** in just a few minutes.

## Features

- **4-bit quantization (NF4 + double quantization)** via `BitsAndBytesConfig` to minimize VRAM usage.
- **LoRA adapter** via `peft` — trains only a small fraction (~1%) of the total parameters.
- **Custom instruction dataset** in chat/messages format, built directly in the notebook.
- **Training** using `SFTTrainer` from the `trl` library.
- **Qualitative evaluation**: comparing model outputs before and after fine-tuning on the same prompts.
- **Save & reload adapter**: saving the LoRA adapter (tens of MB) and reloading it separately from the base model, simulating a deployment scenario.

## Notebook Structure

1. Installing libraries (`transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`)
2. Loading the model in 4-bit precision
3. Baseline: model output before fine-tuning
4. Preparing the instruction dataset
5. LoRA configuration (`LoraConfig`)
6. Training with `SFTTrainer`
7. Evaluation: before vs after fine-tuning
8. Saving and reloading the adapter
9. Summary and discussion

## Requirements

- A Python environment with GPU access (Google Colab's free T4 tier is sufficient)
- Libraries: `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate` (installed automatically in the notebook's first cell)

## How to Run

1. Open `Handson_QLoRA_Finetuning.ipynb` in Google Colab.
2. Set the runtime to **GPU (T4)**: `Runtime > Change runtime type > T4 GPU`.
3. Run all cells sequentially from top to bottom.

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
