# Fine-tuning recipe used in the thesis

These are the exact files behind the reported results. They run inside our fork of
Meta's omnilingual-asr, which holds the fairseq2 training and evaluation code:
[cykyy/omnilingual-asr, branch `rayth1`](https://github.com/cykyy/omnilingual-asr/tree/rayth1).

| File | Purpose |
|---|---|
| `download_regspeech12.py` | Downloads RegSpeech12 from Kaggle (needs a Kaggle account) |
| `prepare_regspeech12.py` | Converts the Kaggle audio + Excel transcripts into fairseq2 manifests (`.tsv` + `.wrd`), resampled to 16 kHz mono |
| `configs/regspeech12-80hr.yaml` | Training config we ran |
| `configs/resolved_training_config.yaml` | The full config fairseq2 actually used, with every default filled in |
| `configs/regspeech12-test.yaml` | Evaluation config for the test split |
| `configs/regspeech12_dataset_card.yaml` | fairseq2 dataset card pointing to the manifests |
| `calculate_epochs.py` | Works out how many epochs 10,000 steps covered (13.6) from the logs |

## Setup

| | |
|---|---|
| Base model | `omniASR_CTC_300M_v2` (325M parameters: wav2vec 2.0 encoder + CTC head) |
| Tokenizer | `omniASR_tokenizer_written_v2` (character level) |
| Fine-tuning | All weights, 10,000 updates (~13.6 epochs over the 80 h training split) |
| Batching | Up to 1,000,000 audio samples (~62 s) per batch, 8 batches per update (~23 clips per update) |
| Optimizer | AdamW, peak learning rate 5e-5, betas (0.9, 0.98), weight decay 0.01 |
| Schedule | Tri-stage: 10% warm-up, 40% hold, 50% decay |
| Precision | bfloat16 mixed precision |
| Hardware | 1 GPU with 16 GB memory, about 6.4 hours |
| Decoding | Greedy CTC, no language model |
| Scoring | Strict WER and UER from fairseq2: no text normalization |

## Reproduce

```bash
git clone -b rayth1 https://github.com/cykyy/omnilingual-asr.git
cd omnilingual-asr
pip install -e .
python prepare_regspeech12.py            # writes dataset/regspeech12/data_manifest
```

Train:

```bash
python -m workflows.recipes.wav2vec2.asr --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml output/
```

Evaluate the fine-tuned checkpoint on the test split (replace the run folder with yours):

```bash
python -m workflows.recipes.wav2vec2.asr.eval --config model.path=output/ws_1.ecbbc39f/checkpoints/step_10000/model --config model.family=wav2vec2_asr --config model.arch=300m_v2 --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml eval_step10000/
```

Evaluate the base model zero-shot:

```bash
python -m workflows.recipes.wav2vec2.asr.eval --config model.name=omniASR_CTC_300M_v2 --config model.path=null --config-file workflows/recipes/wav2vec2/asr/eval/configs/regspeech12-test.yaml eval_baseline/
```

The trained weights are on Hugging Face:
[cykyy/bn-regional-asr-300m](https://huggingface.co/cykyy/bn-regional-asr-300m).
