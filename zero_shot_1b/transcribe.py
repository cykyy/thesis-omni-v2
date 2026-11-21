"""Transcribe Bangla audio files with omniASR_LLM_1B_v2 (zero-shot).

    python zero_shot_1b/transcribe.py path/to/clip1.wav [path/to/clip2.wav ...]
"""

import sys

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

if len(sys.argv) < 2:
    sys.exit(__doc__)

audio_files = sys.argv[1:]
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B_v2")
transcriptions = pipeline.transcribe(audio_files, lang=["ben_Beng"] * len(audio_files), batch_size=1)
for path, text in zip(audio_files, transcriptions):
    print(f"{path}\t{text}")
