# Post-Processing Analysis Plan

## 1. Purpose
After running zero-shot and fine-tuned evaluations using `evaluation.py`, we will perform additional analysis to:  

1. Compute **relative improvement** in WER and CER after fine-tuning.  
2. Extract **representative error examples** for discussion in the thesis.  

This step does not require re-running the ASR pipeline; it operates entirely on the CSV outputs.

---

## 2. Inputs

- `zero_shot_results.csv`: Output CSV from zero-shot evaluation (v2 pretrained weights)  
- `finetuned_results.csv`: Output CSV from evaluation after fine-tuning on RegSpeech12  

Each CSV includes:
file_name, reference, hypothesis, wer, cer


---

## 3. Relative Improvement Analysis

**Goal:** Measure improvement per sample and per dialect.

1. **Merge CSVs** on `file_name`:
   - Add suffixes `_zero` and `_ft` for WER/CER columns.
2. **Compute relative improvement**:
relative_wer = (wer_zero - wer_ft) / wer_zero
relative_cer = (cer_zero - cer_ft) / cer_zero

3. **Aggregate statistics**:
- Overall mean/median/std of relative WER and CER
- Optionally, group by **dialect** (extract from `file_name`) to see per-region improvements.
4. **Output**:
- Save merged CSV with relative improvements.
- Produce summary table of overall and per-dialect improvements for thesis.

---

## 4. Error Category Examples

**Goal:** Highlight typical or interesting transcription errors.

1. **Select a subset** of samples:
- e.g., 20–30 files with high WER/CER
- Include at least one sample per dialect if possible
2. **Compare zero-shot vs fine-tuned**:
- Columns: `file_name`, `reference`, `hypothesis_zero`, `hypothesis_ft`, `wer_zero`, `wer_ft`, `cer_zero`, `cer_ft`
3. **Use in thesis**:
- Illustrate how fine-tuning corrects dialect-specific errors
- Show examples of persistent errors or improvements

---

## 5. Notes

- This analysis can be run **after all evaluation runs** are complete.  
- No additional ASR inference or model loading is required.  
- Outputs (tables/figures) can be referenced in thesis results sections for both **quantitative** (relative improvement) and **qualitative** (error examples) discussions.
