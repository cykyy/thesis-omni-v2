"""
WER/CER Evaluation Code for Omnilingual ASR on Bengali (Regional dialects)
Dataset: RegSpeech12
"""

import pandas as pd
import os
from jiwer import wer, cer
import argparse
from tqdm import tqdm
from datetime import datetime

# Paths
DATASET_ROOT = "/root/.cache/kagglehub/datasets/mdrezuwanhassan/regspeech12/versions/1"
TEST_XLSX = os.path.join(DATASET_ROOT, "test.xlsx")
TEST_AUDIO_DIR = os.path.join(DATASET_ROOT, "test")
RESULTS_DIR = "/root/thesis/results/zero_shot"

# All available dialects
ALL_DIALECTS = [
    "barishal",
    "chittagong", 
    "comilla", 
    "habiganj", 
    "kishoreganj", 
    "narail", 
    "narsingdi", 
    "noakhali", 
    "rangpur", 
    "sandwip", 
    "sylhet", 
    "tangail"
]


def get_output_path(mode: str, dialect: str = "barishal", n_samples: int = None) -> str:
    """Generate timestamped output path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == "smoke":
        filename = f"eval_{dialect}_smoke_{n_samples}samples_{timestamp}.csv"
    else:
        filename = f"eval_{dialect}_full_{timestamp}.csv"
    
    return os.path.join(RESULTS_DIR, filename)


def get_summary_path() -> str:
    """Generate path for all-dialects summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_DIR, f"eval_all_dialects_summary_{timestamp}.csv")


def load_test_data(xlsx_path: str, prefix: str = "test_barishal_") -> pd.DataFrame:
    """Load test.xlsx and filter for specific prefix."""
    df = pd.read_excel(xlsx_path)
    print(f"Total samples in test.xlsx: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Filter for Regional dialect
    df_filtered = df[df['file_name'].str.startswith(prefix)]
    print(f"Samples with prefix '{prefix}': {len(df_filtered)}")
    
    return df_filtered


def save_results(results: list, output_path: str, overall_wer: float, overall_cer: float):
    """Save results to CSV with summary."""
    results_df = pd.DataFrame(results)
    
    # Add summary row
    summary_row = {
        'file_name': '--- OVERALL ---',
        'reference': '',
        'hypothesis': '',
        'wer': overall_wer,
        'cer': overall_cer
    }
    results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_path}")
    
    return results_df


def run_smoke_test(df: pd.DataFrame, audio_dir: str, pipeline, dialect: str = "barishal", n_samples: int = 3):
    """Run a quick smoke test on a few samples."""
    
    print("\n" + "="*60)
    print(f"SMOKE TEST - {dialect.upper()}")
    print("="*60)
    
    # Get first n samples
    samples = df.head(n_samples)
    
    audio_files = []
    references = []
    file_names = []
    
    for _, row in samples.iterrows():
        audio_path = os.path.join(audio_dir, row['file_name'])
        if os.path.exists(audio_path):
            audio_files.append(audio_path)
            references.append(row['transcripts'])
            file_names.append(row['file_name'])
        else:
            print(f"Warning: File not found: {audio_path}")
    
    if not audio_files:
        print("No audio files found!")
        return None, None
    
    # Run inference
    print(f"\nTranscribing {len(audio_files)} files...")
    lang = ["ben_Beng"] * len(audio_files)
    hypotheses = pipeline.transcribe(audio_files, lang=lang, batch_size=1)
    
    # Build results
    results = []
    print("\n" + "-"*60)
    for i, (fname, ref, hyp) in enumerate(zip(file_names, references, hypotheses)):
        sample_wer = wer(ref, hyp)
        sample_cer = cer(ref, hyp)
        
        results.append({
            'file_name': fname,
            'reference': ref,
            'hypothesis': hyp,
            'wer': sample_wer,
            'cer': sample_cer
        })
        
        print(f"\n[{i+1}] File: {fname}")
        print(f"    Reference:  {ref}")
        print(f"    Hypothesis: {hyp}")
        print(f"    WER: {sample_wer:.2%} | CER: {sample_cer:.2%}")
    
    # Overall metrics
    overall_wer = wer(references, hypotheses)
    overall_cer = cer(references, hypotheses)
    
    # Save to CSV
    output_path = get_output_path("smoke", dialect=dialect, n_samples=n_samples)
    save_results(results, output_path, overall_wer, overall_cer)
    
    print("\n" + "-"*60)
    print(f"SMOKE TEST RESULTS - {dialect.upper()} (n={len(audio_files)})")
    print(f"  Overall WER: {overall_wer:.2%}")
    print(f"  Overall CER: {overall_cer:.2%}")
    print("="*60)
    
    return overall_wer, overall_cer


def run_full_evaluation(df: pd.DataFrame, audio_dir: str, pipeline, dialect: str = "barishal", batch_size: int = 8):
    """Run full evaluation on all samples."""
    
    print("\n" + "="*60)
    print(f"FULL EVALUATION - {dialect.upper()}")
    print("="*60)
    
    # Prepare all audio files
    audio_files = []
    references = []
    file_names = []
    missing_files = []
    
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['file_name'])
        if os.path.exists(audio_path):
            audio_files.append(audio_path)
            references.append(row['transcripts'])
            file_names.append(row['file_name'])
        else:
            missing_files.append(row['file_name'])
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files not found")
    
    if not audio_files:
        print("No audio files found!")
        return None, None, None
    
    print(f"Evaluating {len(audio_files)} audio files...")
    
    # Run inference in batches
    lang = ["ben_Beng"] * len(audio_files)
    
    all_hypotheses = []
    for i in tqdm(range(0, len(audio_files), batch_size), desc=f"Transcribing {dialect}"):
        batch_files = audio_files[i:i+batch_size]
        batch_lang = lang[i:i+batch_size]
        batch_hyps = pipeline.transcribe(batch_files, lang=batch_lang, batch_size=batch_size)
        all_hypotheses.extend(batch_hyps)
    
    # Calculate overall metrics
    overall_wer = wer(references, all_hypotheses)
    overall_cer = cer(references, all_hypotheses)
    
    # Build per-sample results
    results = []
    for fname, ref, hyp in zip(file_names, references, all_hypotheses):
        sample_wer = wer(ref, hyp)
        sample_cer = cer(ref, hyp)
        results.append({
            'file_name': fname,
            'reference': ref,
            'hypothesis': hyp,
            'wer': sample_wer,
            'cer': sample_cer
        })
    
    # Save to CSV
    output_path = get_output_path("full", dialect=dialect)
    results_df = save_results(results, output_path, overall_wer, overall_cer)
    
    # Print summary
    print("\n" + "-"*60)
    print(f"FULL EVALUATION RESULTS - {dialect.upper()}")
    print("-"*60)
    print(f"  Total samples:  {len(audio_files)}")
    print(f"  Overall WER:    {overall_wer:.2%}")
    print(f"  Overall CER:    {overall_cer:.2%}")
    print(f"  Mean WER:       {results_df['wer'].iloc[:-1].mean():.2%}")
    print(f"  Median WER:     {results_df['wer'].iloc[:-1].median():.2%}")
    print(f"  Std WER:        {results_df['wer'].iloc[:-1].std():.2%}")
    print(f"  Mean CER:       {results_df['cer'].iloc[:-1].mean():.2%}")
    print(f"  Median CER:     {results_df['cer'].iloc[:-1].median():.2%}")
    print(f"  Std CER:        {results_df['cer'].iloc[:-1].std():.2%}")
    print("="*60)
    
    return results_df, overall_wer, overall_cer


def load_pipeline():
    """Load ASR pipeline once."""
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    print("Loading ASR pipeline...")
    return ASRInferencePipeline(model_card="omniASR_LLM_1B_v2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WER/CER on Bengali ASR")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                        help="Run smoke test or full evaluation")
    parser.add_argument("--dialect", type=str, default="barishal",
                        choices=ALL_DIALECTS + ["all"],
                        help="Dialect to evaluate, or 'all' for all dialects")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of samples for smoke test")
    parser.add_argument("--batch_size", type=int, default=7,
                        help="Batch size for full evaluation")
    
    args = parser.parse_args()
    
    # Load pipeline once
    pipeline = load_pipeline()
    
    # Determine which dialects to process
    dialects_to_process = ALL_DIALECTS if args.dialect == "all" else [args.dialect]
    
    summary_results = []
    
    for dialect in dialects_to_process:
        print(f"\n{'#'*60}")
        print(f"# Processing dialect: {dialect.upper()}")
        print(f"{'#'*60}")
        
        df = load_test_data(TEST_XLSX, prefix=f"test_{dialect}_")
        
        if len(df) == 0:
            print(f"No samples found for {dialect}, skipping...")
            continue
        
        if args.mode == "smoke":
            overall_wer, overall_cer = run_smoke_test(
                df, TEST_AUDIO_DIR, pipeline,
                dialect=dialect, n_samples=args.n_samples
            )
        else:  # full
            _, overall_wer, overall_cer = run_full_evaluation(
                df, TEST_AUDIO_DIR, pipeline,
                dialect=dialect, batch_size=args.batch_size
            )
        
        if overall_wer is not None:
            summary_results.append({
                'dialect': dialect,
                'n_samples': len(df) if args.mode == "full" else min(args.n_samples, len(df)),
                'wer': overall_wer,
                'cer': overall_cer
            })
    
    # Save summary if multiple dialects were processed
    if len(summary_results) > 1:
        summary_df = pd.DataFrame(summary_results)
        summary_df = summary_df.sort_values('wer')
        
        # Add average row
        avg_row = {
            'dialect': '--- AVERAGE ---',
            'n_samples': summary_df['n_samples'].sum(),
            'wer': summary_df['wer'].mean(),
            'cer': summary_df['cer'].mean()
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        summary_path = get_summary_path()
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*60)
        print("ALL DIALECTS SUMMARY (sorted by WER)")
        print("="*60)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
