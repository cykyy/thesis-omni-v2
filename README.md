# Fine-Tuning Multilingual ASR for Bengali Regional Dialects: A Zero-Shot and Adaptation Study

BSc thesis in Computer Science and Engineering, American International University-Bangladesh
(AIUB), defended in 2026.

**Authors:** Rayhan Mia, Sudipta Shikha Ausmita, Prajnadip Das Samya, Farduna Tajkia
**Supervisor:** Aminun Nahar, Department of Computer Science, AIUB

| | |
|---|---|
| Fine-tuned model | [cykyy/bn-regional-asr-300m](https://huggingface.co/cykyy/bn-regional-asr-300m) on Hugging Face |
| Training and evaluation code | [cykyy/omnilingual-asr, branch `rayth1`](https://github.com/cykyy/omnilingual-asr/tree/rayth1) (our fork of Meta's Omnilingual ASR) |
| Recipe, configs and results | this repository |

## Summary

Speech recognition works well for standard Bangla but poorly for the regional dialects that
millions of people actually speak. We took Meta's multilingual
[Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) model
`omniASR_CTC_300M_v2`, measured how it performs on 12 Bangladeshi dialects without any
adaptation (zero-shot), then fine-tuned it on 80 hours of dialect speech from
[RegSpeech12](https://arxiv.org/abs/2510.24096).

Fine-tuning cut the word error rate on the test set from **97.93% to 72.97%** and the
character-level error rate from **81.74% to 35.44%**, with gains in every one of the 12
dialects. The much larger character-level gain shows the model learned to hear dialect speech;
we attribute much of the remaining word error to dialect spelling variation and the absence of
a language model.

## Research questions

1. How badly does a pre-trained multilingual ASR model perform on spontaneous regional Bangla
   speech without adaptation?
2. How much does supervised fine-tuning on RegSpeech12 improve word-level and character-level
   transcription?
3. What errors remain after fine-tuning, and what do they tell us about dialectal Bangla ASR?

## Data

[RegSpeech12](https://arxiv.org/abs/2510.24096) (Hassan et al., 2025) has about 100 hours of
spontaneous speech from 12 regions of Bangladesh: Barishal, Chittagong, Comilla, Habiganj,
Kishoreganj, Narail, Narsingdi, Noakhali, Rangpur, Sandwip, Sylhet and Tangail.

| Split | Clips | Audio |
|---|---:|---:|
| Train | 17,049 | ~80 h |
| Validation | 2,132 | ~10 h |
| Test | 2,132 | ~10 h |

Transcripts are written in each dialect's own spelling, not standard Bangla. The test split is
unbalanced: Sylhet has 762 clips (36%), Noakhali only 28.

## Method

- **Model:** `omniASR_CTC_300M_v2`, a 325M-parameter wav2vec 2.0 encoder with a CTC output
  layer over characters, pre-trained by Meta and supporting 1,600+ languages.
- **Fine-tuning:** all weights, 10,000 updates (~13.6 epochs) with AdamW (peak learning rate
  5e-5, tri-stage schedule) in bfloat16, on one 16 GB GPU in about 6.4 hours. Built on Meta's
  fairseq2 training recipe.
- **Evaluation:** test split, greedy CTC decoding (no language model), word error rate (WER)
  and unit (character) error rate (UER) with strict scoring: no text normalization, so
  punctuation and spelling differences count as errors.

Full settings and commands to reproduce: [`training/`](training/).

## Results

### Overall (test split, 2,132 clips)

| Model | WER | UER | CTC loss |
|---|---:|---:|---:|
| Zero-shot `omniASR_CTC_300M_v2` | 97.93% | 81.74% | 1,482.45 |
| **Fine-tuned (ours)** | **72.97%** | **35.44%** | **393.83** |
| Change | −24.96 pts (−25.5%) | −46.30 pts (−56.6%) | −73.4% |

Re-scored after the thesis while ignoring punctuation (every punctuation mark replaced by a
space), the fine-tuned WER is 68.01%. Logs: [`results/300m_ctc/metrics/`](results/300m_ctc/metrics/).

### Per dialect

| Dialect | Test clips | Zero-shot WER | Fine-tuned WER | Fine-tuned CER |
|---|---:|---:|---:|---:|
| Tangail | 127 | 94.98% | **46.37%** | 15.34% |
| Comilla | 32 | 89.08% | **48.19%** | 15.87% |
| Habiganj | 117 | 99.47% | **61.91%** | 24.00% |
| Narail | 186 | 95.72% | **65.80%** | 29.19% |
| Narsingdi | 137 | 99.73% | **71.42%** | 30.57% |
| Barishal | 101 | 95.85% | **72.89%** | 36.09% |
| Sylhet | 762 | 98.74% | **74.88%** | 38.66% |
| Rangpur | 130 | 96.97% | **76.24%** | 39.03% |
| Sandwip | 131 | 99.78% | **79.15%** | 40.26% |
| Chittagong | 176 | 97.18% | **82.20%** | 38.57% |
| Noakhali | 28 | 99.05% | **82.29%** | 44.62% |
| Kishoreganj | 205 | 99.60% | **87.05%** | 47.29% |
| **All** | **2,132** | **97.93%** | **72.97%** | **35.49%** |

Per-dialect scores come from re-scoring the released test transcripts; each clip's dialect is
taken from its file name. CER here counts characters including spaces, so the overall 35.49%
differs slightly from fairseq2's 35.44% UER. Data:
[`results/300m_ctc/per_dialect.csv`](results/300m_ctc/per_dialect.csv); every clip's reference
and both model outputs: [`results/300m_ctc/test_transcripts.tsv`](results/300m_ctc/test_transcripts.tsv).

### What we found

- **Without adaptation the model is unusable on dialect speech:** nearly every word is wrong in
  every dialect (89–100% WER).
- **Fine-tuning helps every dialect, but unevenly.** Tangail and Comilla drop below 50% WER;
  Kishoreganj, Noakhali and Chittagong stay above 80%.
- **Characters improve much more than words** (−56.6% vs −25.5% relative). The model now gets most
  characters right, but a single wrong letter makes the whole word wrong. Dialect words have
  no standard spelling, and greedy decoding has no language model to pick valid words.
- **Small dialects are uncertain.** Noakhali (28 clips) and Comilla (32) are too small for firm
  per-dialect conclusions.

## Limitations

- Greedy decoding without a language model limits word-level accuracy.
- One model for all 12 dialects, trained on 80 hours; some dialects have little data.
- One training run, without repeated seeds or confidence intervals.
- Tested only on RegSpeech12; other recording conditions are untested.

## Follow-up work (in progress)

- **n-gram language model decoding:** beam search with a KenLM model trained on the training
  transcripts. Code: [`scripts/lm/` in our fork](https://github.com/cykyy/omnilingual-asr/tree/rayth1/scripts/lm).
- **Stronger comparisons:** other public models (Whisper, MMS, IndicWav2Vec) scored with the same
  scorer, and per-dialect confidence intervals.

## Early experiment: zero-shot 1B model

Before choosing the 300M model we tested the larger `omniASR_LLM_1B_v2` (wav2vec 2.0 encoder
+ LLaMA decoder) zero-shot on the same test split. It does better than the 300M model zero-shot
(**88.17% WER**, 51.20% CER) but much worse than our fine-tuned 300M model, which beats it in
every dialect. Code and per-clip results: [`zero_shot_1b/`](zero_shot_1b/) and
[`results/1b_llm_zero_shot/`](results/1b_llm_zero_shot/).

## Repository layout

```
├── training/                 # thesis fine-tuning recipe: data prep, configs, reproduce steps
├── results/
│   ├── 300m_ctc/             # main thesis results: metrics, per-dialect table, transcripts
│   └── 1b_llm_zero_shot/     # early 1B zero-shot experiment, per dialect
├── zero_shot_1b/             # scripts for the 1B zero-shot experiment
├── docs/                     # original experiment plan, a bug we fixed in the WER calculator
└── archive/                  # abandoned 1B fine-tuning attempts (never produced results)
```

## Acknowledgements

We thank our supervisor Aminun Nahar for her guidance. The base model and training framework
are Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) and
[fairseq2](https://github.com/facebookresearch/fairseq2). The dataset is
[RegSpeech12](https://www.kaggle.com/datasets/mdrezuwanhassan/regspeech12) by Hassan et al.

> **Note (Nov 2026):** this repository first held only the early 1B experiment. It now covers
> the whole thesis. An earlier summary file labelled the unweighted mean of the 12 dialects
> (85.3% WER) as the 1B model's overall average; the figure over all clips is 88.17%.
