# Archive

Earlier attempts, kept for the record. **None of these produced any reported result.**

| Path | What it was | Why it is here |
|---|---|---|
| `1b_finetune_setup/` | Data conversion (Parquet, manifests), dataset card, config and launcher for fine-tuning the **1B LLM** model with the fairseq2 recipe | The 1B fine-tuning was set aside; the thesis fine-tuned the 300M CTC model instead (see [`../training/`](../training/)). Paths inside assume our old server layout (`/root/thesis`). |
| `fine_tune.py`, `fine_tune_config.yaml` | A hand-written PyTorch training loop for the 1B model | It imports `create_model` from `omnilingual_asr.models.wav2vec2_llama.factory`, which does not exist in the pinned omnilingual-asr version, so it never ran. We switched to the official fairseq2 recipe. |
| `finetune_config_old.yaml` | First version of the 1B fairseq2 config | Replaced by `1b_finetune_setup/finetune_config.yaml`. |
