# Early experiment: zero-shot `omniASR_LLM_1B_v2`

Before the main thesis experiments (December 2025) we tested Meta's larger 1B model
(wav2vec 2.0 encoder + LLaMA decoder) on the RegSpeech12 test split without any training,
with language code `ben_Beng`. Results per dialect and per clip:
[`../results/1b_llm_zero_shot/`](../results/1b_llm_zero_shot/).

| | WER | CER |
|---|---:|---:|
| All 2,132 test clips (pooled) | 88.17% | 51.20% |
| Unweighted mean of the 12 dialects | 85.33% | 49.26% |

Scored with [jiwer](https://github.com/jitsi/jiwer) defaults (no text normalization). The
pooled figure is the standard one; the mean gives tiny dialects such as Noakhali (28 clips)
the same weight as Sylhet (762).

## Run

```bash
pip install -r zero_shot_1b/requirements.txt
python training/download_regspeech12.py
python zero_shot_1b/zero_shot_eval.py --mode smoke --dialect barishal --n_samples 3
python zero_shot_1b/zero_shot_eval.py --mode full --dialect all
```

Set `REGSPEECH12_ROOT` if the dataset is not in kagglehub's default cache folder.
`transcribe.py` transcribes any audio files you pass it; `check_gpu.py` lists the GPUs.
