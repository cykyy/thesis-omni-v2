from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B_v2")
audio_files = ["/root/thesis/data/test_barishal_0001.wav"]
lang = ["ben_Beng"]
transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=1)
print(transcriptions[0])