# Thesis Experiment Plan Overview

## Title
A Comparative Study of Zero-Shot vs Fine-Tuned Omnilingual ASR 1B v2 on Regional Bengali Dialects

---

## 1. Experiment Scope

- Focus on **Omnilingual ASR LLM 1B v2**.
- Evaluate **zero-shot performance** on dialectally diverse Bengali speech (RegSpeech12).
- Fine-tune the model using **train split of RegSpeech12**.
- Compare performance before and after fine-tuning on the **same test split**.
- Optional, minimal-effort by-products included for richer analysis.

---

## 2. Dataset

**RegSpeech12**: A spontaneous Bengali speech corpus from 12 regions across Bangladesh (~100 hours, 21,313 clips, 394+ speakers).  
- Train: 17,049 clips (~80 hours)  
- Validation: 2,132 clips (~10 hours)  
- Test: 2,132 clips (~10 hours)  
- Covers phonological, lexical, and morphological dialectal variation.  
- Recordings are natural, free-form monologues.

---

## 3. Core Experimental Flow

### Step 1: Zero-Shot Evaluation
- Model: `omniASR_LLM_1B_v2` (pretrained, no training)
- Data: **Test split only**
- Metrics:
  - WER
  - CER
- Output: Baseline performance of v2 without any dialect-specific adaptation.

### Step 2: Fine-Tuning
- Model: Initialize from v2 pretrained weights
- Train: **Train split (~80 hours)**
- Validation: **Validation split (~10 hours)**
- Monitor: Training loss and validation WER/CER
- Goal: Adapt the model to regional Bengali speech

### Step 3: Post-Training Evaluation
- Model: Fine-tuned v2
- Data: **Same test split as zero-shot**
- Metrics:
  - WER
  - CER
- Output: Performance of fine-tuned model
- Compare directly with zero-shot baseline

---

## 4. Low-Effort By-Products

1. **Per-Dialect Metrics**
   - Compute WER/CER for each of the 12 regions
   - Compare zero-shot vs fine-tuned results

2. **Relative Improvement**
   - Compute percentage improvement:
     ```
     Relative WER Reduction = (WER_zero - WER_finetuned) / WER_zero
     ```
   - Provides normalized view of fine-tuning effect

3. **Error Category Examples**
   - Select a small sample (20-30 clips)
   - Display:
     - Ground truth
     - Zero-shot transcription
     - Fine-tuned transcription
   - Highlight dialectal words corrected or phonological errors fixed

---

## 5. Key Notes

- **Test set is identical** for zero-shot and fine-tuned evaluation to ensure fair comparison.
- Only v2 is evaluated; v1 is not included to maintain focus on zero-shot vs fine-tuned analysis.
- Any observations of dialect-specific failures or improvements will be discussed in the thesis.
- Environmental noise, spontaneous speech, and regional variations are considered part of the challenge.

---

This document can serve as the **main roadmap** for the experiments and be stored in the thesis repository.
