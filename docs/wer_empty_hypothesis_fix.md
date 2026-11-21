> Found and fixed during the thesis training runs. The fix (7 lines) is in
> [`workflows/recipes/wav2vec2/asr/wer_calculator.py`](https://github.com/cykyy/omnilingual-asr/blob/rayth1/workflows/recipes/wav2vec2/asr/wer_calculator.py)
> on the `rayth1` branch of our omnilingual-asr fork. File paths below refer to that repository.

# Bug Report: WER Calculator Crashes on Empty CTC Hypotheses

## Summary

The `WerCalculator` in `workflows/recipes/wav2vec2/asr/wer_calculator.py` crashes with a `ValueError` when the CTC model produces all-blank outputs during validation. This commonly occurs early in training before the model has learned to produce meaningful tokens.

## Environment

- **Repository**: omnilingual-asr
- **File**: `workflows/recipes/wav2vec2/asr/wer_calculator.py`
- **Function**: `_generate_hypotheses()` (line 178-194)
- **Dependency**: fairseq2 `pad_seqs()` function

## Error Message

```
ValueError: All lengths in `seq_lens` must be greater than or equal to 1, but the length at index 0 is 0 instead.
```

## Full Stack Trace

```
File "/root/omnilingual-asr/workflows/recipes/wav2vec2/asr/wer_calculator.py", line 194, in _generate_hypotheses
    return pad_seqs(hyp_seqs, pad_value=self._pad_idx)
File "/root/omnilingual-asr/venv/lib/python3.10/site-packages/fairseq2/nn/utils/padding.py", line 49, in pad_seqs
    seqs_layout = BatchLayout.of(padded_seqs, data["seq_lens"])
File "/root/omnilingual-asr/venv/lib/python3.10/site-packages/fairseq2/nn/batch_layout.py", line 172, in of
    return BatchLayout(shape, seq_lens, packed=packed, device=batch.device)
File "/root/omnilingual-asr/venv/lib/python3.10/site-packages/fairseq2/nn/batch_layout.py", line 134, in __init__
    raise ValueError(
ValueError: All lengths in `seq_lens` must be greater than or equal to 1, but the length at index 0 is 0 instead.
```

## Steps to Reproduce

1. Configure training with early validation:
   ```yaml
   regime:
     validate_after_n_steps: 500  # Early validation
     score_metric: "wer"
   ```

2. Start training on a new dataset (model not pre-adapted to the data)

3. Training crashes at first validation step when WER is computed

## Root Cause Analysis

In `_generate_hypotheses()`, the code performs CTC greedy decoding:

```python
def _generate_hypotheses(self, logits: Tensor, logit_layout: BatchLayout):
    hyp_seqs = []

    for logits, logits_len in zip(logits, logit_layout.seq_lens):
        # Get greedy prediction
        hyp_seq = logits[:logits_len].argmax(-1).unique_consecutive()

        # Remove blank tokens - THIS CAN RESULT IN EMPTY SEQUENCE
        hyp_seq = hyp_seq[hyp_seq != self._blank_label]

        hyp_seqs.append(hyp_seq)

    # CRASHES HERE if any hyp_seq has length 0
    return pad_seqs(hyp_seqs, pad_value=self._pad_idx)
```

**The issue:** When the model outputs all blank tokens (common early in training), removing blanks produces an empty tensor (length 0). The `pad_seqs()` function from fairseq2 requires all sequences to have length >= 1.

### Why This Happens

1. **CTC blank tokens**: CTC models use a special "blank" token to represent no output at a given time step
2. **Early training**: Before the model learns, it may output mostly/all blanks
3. **Greedy decoding**: `argmax(-1).unique_consecutive()` followed by blank removal can yield empty sequences
4. **No empty check**: The code doesn't handle the edge case of empty hypotheses

### Why Default Config Avoids This

Meta's default configuration sets:
```python
validate_after_n_steps=10_000  # Very late validation
num_steps=20_000
```

By step 10,000, the model typically produces non-blank outputs, so the issue doesn't manifest. However, users who want earlier validation (for faster feedback or smaller datasets) will hit this bug.

## Proposed Fix

Add a check for empty sequences and handle them gracefully:

```python
def _generate_hypotheses(self, logits: Tensor, logit_layout: BatchLayout):
    import torch

    hyp_seqs = []

    for logits, logits_len in zip(logits, logit_layout.seq_lens):
        hyp_seq = logits[:logits_len].argmax(-1).unique_consecutive()
        hyp_seq = hyp_seq[hyp_seq != self._blank_label]

        # Handle empty sequences (model outputs all blanks)
        # Add a single PAD token to avoid pad_seqs error
        if hyp_seq.numel() == 0:
            hyp_seq = torch.tensor([self._pad_idx], device=logits.device)

        hyp_seqs.append(hyp_seq)

    return pad_seqs(hyp_seqs, pad_value=self._pad_idx)
```

### Why This Fix is Correct

1. **Empty hypothesis = silence prediction**: The model predicting "nothing" is valid behavior
2. **WER calculation**: An empty/PAD hypothesis compared to actual text will yield ~100% WER (expected for untrained model)
3. **No behavior change for normal cases**: Only affects edge case of empty outputs
4. **Graceful degradation**: Training continues instead of crashing

## Alternative Solutions

### Option 1: Skip WER for empty batches
```python
if all(seq.numel() == 0 for seq in hyp_seqs):
    log.warning("All hypotheses are empty, skipping WER calculation")
    return None  # Would need changes to caller
```

### Option 2: Use a special "empty" token
```python
EMPTY_TOKEN_IDX = tokenizer.vocab_info.unk_idx  # or a dedicated token
if hyp_seq.numel() == 0:
    hyp_seq = torch.tensor([EMPTY_TOKEN_IDX], device=logits.device)
```

### Option 3: Document the limitation
Add to documentation that `validate_after_n_steps` should be set high enough for the model to produce outputs. (Not recommended - crashes are poor UX)

## Impact

- **Severity**: Medium (crashes training, but has workaround)
- **Workaround**: Set `validate_after_n_steps` >= 10000 or disable WER metric
- **Users affected**: Anyone doing early validation or fine-tuning on small/new datasets

## Related Files

- `workflows/recipes/wav2vec2/asr/wer_calculator.py` - Bug location
- `workflows/recipes/wav2vec2/asr/default_config.py` - Default config (avoids bug)
- `fairseq2/nn/utils/padding.py` - `pad_seqs()` function that raises the error

## Suggested GitHub Issue Title

```
[Bug] WER calculator crashes on empty CTC hypotheses during early validation
```

## Suggested GitHub Issue Labels

- `bug`
- `wav2vec2`
- `asr`
- `training`

---

## Patch File

For convenience, here's a patch that can be applied:

```diff
diff --git a/workflows/recipes/wav2vec2/asr/wer_calculator.py b/workflows/recipes/wav2vec2/asr/wer_calculator.py
index abc1234..def5678 100644
--- a/workflows/recipes/wav2vec2/asr/wer_calculator.py
+++ b/workflows/recipes/wav2vec2/asr/wer_calculator.py
@@ -175,6 +175,8 @@ class WerCalculator:

     def _generate_hypotheses(
         self, logits: Tensor, logit_layout: BatchLayout
     ) -> tuple[Tensor, BatchLayout]:
+        import torch
+
         hyp_seqs = []

         # Get the greedy token (i.e. unit) output of the model.
@@ -185,6 +187,11 @@ class WerCalculator:
             # (S - blank)
             hyp_seq = hyp_seq[hyp_seq != self._blank_label]

+            # Handle empty sequences (model outputs all blanks)
+            # Add a single PAD token to avoid pad_seqs error
+            if hyp_seq.numel() == 0:
+                hyp_seq = torch.tensor([self._pad_idx], device=logits.device)
+
             hyp_seqs.append(hyp_seq)

         # (N, S), (N, S)
```

---

*Report prepared for submission to: https://github.com/facebookresearch/omnilingual-asr/issues*
