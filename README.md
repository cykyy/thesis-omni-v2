# Thesis: Zero-Shot vs Fine-Tuned Bengali ASR on RegSpeech12

This repository contains the code, data organization, and analysis for evaluating Omnilingual ASR on regional Bengali dialects. The study compares **zero-shot performance** of the `omniASR_LLM_1B_v2` model with **fine-tuned performance** on the RegSpeech12 dataset.

---

## Repository Structure

```
thesis_repo/
├── README.md
├── requirements.txt              # Python dependencies
├── data/
│   └── regspeech12/              # Dataset splits and metadata
├── src/
│   ├── evaluation.py             # WER/CER evaluation script
│   ├── fine_tune.py              # Fine-tuning script (separate)
│   └── post_process.py           # Relative improvement & error examples
├── results/
│   ├── zero_shot/                # Zero-shot evaluation CSVs
│   ├── fine_tuned/               # Fine-tuned evaluation CSVs
│   └── post_process/             # Relative improvements and error examples
├── logs/                         # Optional training/evaluation logs
└── docs/
    └── analysis_plan.md          # Post-processing and analysis plan
```

---

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Omnilingual ASR:

```bash
pip install omnilingual-asr
```

3. Place the RegSpeech12 dataset under `data/regspeech12/` with predefined train/val/test splits.

---

## Usage

### 1. Evaluation

Run zero-shot or fine-tuned evaluation:

```bash
python src/evaluation.py --mode smoke --n_samples 3   # Quick smoke test
python src/evaluation.py --mode full --batch_size 8  # Full evaluation
```

* Outputs CSVs to `results/zero_shot/` or `results/fine_tuned/` depending on the model used.
* CSV includes `file_name`, `reference`, `hypothesis`, `wer`, and `cer`.

### 2. Fine-Tuning

Run `src/fine_tune.py` to fine-tune the model on RegSpeech12 train/val splits.

* Produces fine-tuned checkpoint for later evaluation.

### 3. Post-Processing

After both zero-shot and fine-tuned evaluations:

```bash
python src/post_process.py
```

* Computes **relative WER/CER improvement**.
* Extracts **error examples** to highlight dialect-specific issues.
* Saves outputs to `results/post_process/`.

---

## Notes

* Only `v2` models are used in evaluations (`omniASR_LLM_1B_v2` recommended).
* Each dialect can be evaluated separately; scripts can be run multiple times with different prefixes.
* All scripts are independent and use CSV outputs for analysis; no re-inference is needed for post-processing.

---

## References

* **Omnilingual ASR:** Open-source multilingual speech recognition supporting 1600+ languages.
* **RegSpeech12 Dataset:** Large-scale Bengali spontaneous speech corpus
