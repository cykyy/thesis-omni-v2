# Archive

Early attempts kept for the record. **None of these were used for any reported result.**

| File | What it was | Why it is here |
|---|---|---|
| `fine_tune.py`, `fine_tune_config.yaml` | A hand-written PyTorch training loop for `omniASR_LLM_1B_v2` | It imports `create_model` from `omnilingual_asr.models.wav2vec2_llama.factory`, which does not exist in the pinned omnilingual-asr version, so it never ran. We switched to the official fairseq2 recipe instead. |
| `finetune_config_old.yaml` | First version of the fairseq2 recipe config for the 1B model | Replaced by `configs/finetune_config.yaml`. |
