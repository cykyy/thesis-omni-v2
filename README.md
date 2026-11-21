# Zero-shot Omnilingual ASR on regional Bangla dialects (RegSpeech12)

This repository is the **first stage of our thesis** on speech recognition for regional
Bangla dialects (December 2025). It measures how well Meta's
[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) model
`omniASR_LLM_1B_v2` transcribes the 12 dialects of the
[RegSpeech12](https://arxiv.org/abs/2510.24096) corpus **without any training on them**
(zero-shot), dialect by dialect.

> **Where the thesis went next.** We then fine-tuned the smaller `omniASR_CTC_300M_v2`
> model on RegSpeech12's 80-hour training split. On the same test split it reaches
> **72.97% WER and 35.44% UER**, down from 97.93% WER zero-shot for that model.
>
> - Code, configs and evaluation logs: [cykyy/omnilingual-asr, branch `rayth1`](https://github.com/cykyy/omnilingual-asr/tree/rayth1)
> - Released model: [cykyy/bn-regional-asr-300m on Hugging Face](https://huggingface.co/cykyy/bn-regional-asr-300m)

## Zero-shot results (`omniASR_LLM_1B_v2`, RegSpeech12 test split)

All 2,132 test clips, transcribed with language code `ben_Beng`. Scored with
[jiwer](https://github.com/jitsi/jiwer) defaults: no text normalization, so punctuation
and spelling differences count as errors.

| Dialect | Clips | WER | CER |
|---|---:|---:|---:|
| Tangail | 127 | 52.6% | 20.5% |
| Comilla | 32 | 56.0% | 22.0% |
| Narail | 186 | 77.8% | 41.5% |
| Barishal | 101 | 83.7% | 49.8% |
| Narsingdi | 137 | 84.1% | 45.2% |
| Habiganj | 117 | 85.4% | 44.4% |
| Rangpur | 130 | 91.8% | 57.9% |
| Sylhet | 762 | 93.2% | 54.5% |
| Sandwip | 131 | 95.0% | 60.3% |
| Chittagong | 176 | 97.1% | 56.0% |
| Kishoreganj | 205 | 98.5% | 66.3% |
| Noakhali | 28 | 108.6% | 72.7% |
| **All clips (pooled)** | **2,132** | **88.2%** | **51.2%** |
| Mean of the 12 dialects | | 85.3% | 49.3% |

- **Pooled** is the standard figure: every clip scored together.
- **Mean of the 12 dialects** gives each dialect equal weight. It is lower because the test
  split is unbalanced: Sylhet alone is 762 clips (36%), Noakhali only 28.
- WER above 100% (Noakhali) means the model produced more wrong words than the reference has.
- Tangail and Comilla speech is handled far better than Chittagong, Sylhet, Sandwip or
  Noakhali speech.

Per-clip references, hypotheses and scores are in [`results/zero_shot/`](results/zero_shot/).

> **Correction (Nov 2026):** an earlier version of the summary file labelled the
> unweighted mean of the 12 dialects (85.3% WER) as the overall average. The pooled
> figure over all clips is 88.2% WER / 51.2% CER. The per-dialect numbers are unchanged.

## Reproduce

1. Install dependencies (Python 3.10 or newer; a CUDA GPU is strongly recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Download RegSpeech12 from Kaggle (needs a Kaggle account):

   ```bash
   python src/corpus.py
   ```

3. Run the evaluation:

   ```bash
   python src/zero_shot_eval.py --mode smoke --dialect barishal --n_samples 3   # quick check
   python src/zero_shot_eval.py --mode full --dialect all                       # all 12 dialects
   ```

   Set `REGSPEECH12_ROOT` if the dataset is not in kagglehub's default cache folder, and
   `RESULTS_DIR` to change where results are written (default `results/zero_shot`).

`src/main.py` transcribes any audio files you pass it, and `src/gtest.py` prints the
available GPUs.

## Repository layout

```
├── src/
│   ├── zero_shot_eval.py      # per-dialect zero-shot evaluation (produced results/)
│   ├── corpus.py              # downloads RegSpeech12 via kagglehub
│   ├── create_manifest.py     # RegSpeech12 -> fairseq2 manifest files
│   ├── main.py                # transcribe a few files
│   └── gtest.py               # GPU check
├── convert_to_parquet.py      # RegSpeech12 -> fairseq2 Parquet format
├── run_finetune.sh            # 1B fine-tuning launcher (not completed, see below)
├── configs/finetune_config.yaml
├── cards/datasets/            # fairseq2 dataset card for the Parquet data
├── results/zero_shot/         # per-dialect CSVs + summary
├── docs/                      # original experiment plan and notes
└── archive/                   # early attempts, never used for results
```

## Status of the fine-tuning setup in this repository

`run_finetune.sh`, `configs/finetune_config.yaml`, `convert_to_parquet.py` and
`cards/` prepare a fine-tuning run of the **1B** model with the official fairseq2 recipe.
That run was not completed; the thesis fine-tuned the 300M CTC model instead (links
above). `archive/` holds an earlier
hand-written training script that never worked; see [`archive/README.md`](archive/README.md).

## Data

RegSpeech12 (Hassan et al., 2025, [arXiv:2510.24096](https://arxiv.org/abs/2510.24096);
[Kaggle](https://www.kaggle.com/datasets/mdrezuwanhassan/regspeech12)) has about 100 hours
of spontaneous speech from 12 regions of Bangladesh: 17,049 training, 2,132 validation and
2,132 test clips. The audio is not included here.
