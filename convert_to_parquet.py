"""
RegSpeech12 → Parquet Conversion (Memory Efficient)

Processes in batches to avoid OOM on large datasets.

Usage:
    python convert_to_parquet.py --output_dir /root/thesis/data/regspeech12_parquet
    python convert_to_parquet.py --batch_size 100  # Smaller batch if still OOM
"""

import os
import io
import gc
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm


# Constants
DATASET_ROOT = "/root/.cache/kagglehub/datasets/mdrezuwanhassan/regspeech12/versions/1"
SAMPLE_RATE = 16000
CORPUS_NAME = "regspeech12"
LANGUAGE_CODE = "ben_Beng"
ROW_GROUP_SIZE = 100


def binary_to_list_int8(audio_bytes: bytes) -> list:
    """Convert binary audio bytes to list of int8."""
    return list(np.frombuffer(audio_bytes, dtype=np.int8))


def wav_to_flac_bytes(wav_path: str) -> tuple[bytes, int]:
    """Read WAV file and convert to FLAC bytes."""
    audio, sr = sf.read(wav_path, dtype='float32')
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    audio_size = len(audio)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format='FLAC')
    flac_bytes = buffer.getvalue()
    
    return flac_bytes, audio_size


def process_batch(rows: list, audio_dir: str, target_split_name: str) -> tuple[list, int, int]:
    """Process a batch of rows and return records."""
    records = []
    skipped = 0
    total_audio_size = 0
    
    for row in rows:
        audio_path = os.path.join(audio_dir, row['file_name'])
        
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        try:
            flac_bytes, audio_size = wav_to_flac_bytes(audio_path)
            audio_bytes_list = binary_to_list_int8(flac_bytes)
            
            records.append({
                'text': str(row['transcripts']),
                'audio_bytes': audio_bytes_list,
                'audio_size': audio_size,
                'corpus': CORPUS_NAME,
                'split': target_split_name,
                'language': LANGUAGE_CODE,
            })
            total_audio_size += audio_size
            
        except Exception as e:
            print(f"Error: {row['file_name']}: {e}")
            skipped += 1
    
    return records, skipped, total_audio_size


def write_parquet_batch(records: list, output_file: Path, append: bool = False):
    """Write a batch of records to parquet file."""
    if not records:
        return
    
    table = pa.table({
        'text': pa.array([r['text'] for r in records], type=pa.string()),
        'audio_bytes': pa.array([r['audio_bytes'] for r in records], type=pa.list_(pa.int8())),
        'audio_size': pa.array([r['audio_size'] for r in records], type=pa.int64()),
        'corpus': pa.array([r['corpus'] for r in records]).dictionary_encode(),
        'split': pa.array([r['split'] for r in records]).dictionary_encode(),
        'language': pa.array([r['language'] for r in records]).dictionary_encode(),
    })
    
    if append and output_file.exists():
        # Read existing and concatenate
        existing = pq.read_table(output_file)
        table = pa.concat_tables([existing, table])
    
    pq.write_table(table, output_file, row_group_size=ROW_GROUP_SIZE, compression='snappy')


def process_split(
    xlsx_path: str,
    audio_dir: str,
    source_split_name: str,
    target_split_name: str,
    output_dir: str,
    batch_size: int = 200,
) -> dict:
    """Process a single split in batches."""
    print(f"\n{'='*60}")
    print(f"Processing: {source_split_name} → {target_split_name}")
    print(f"{'='*60}")
    
    df = pd.read_excel(xlsx_path)
    total_samples = len(df)
    print(f"Total samples: {total_samples}")
    print(f"Processing in batches of {batch_size}")
    
    if total_samples == 0:
        return {'split': target_split_name, 'samples': 0, 'skipped': 0, 'hours': 0}
    
    # Output directory
    parquet_dir = Path(output_dir) / f"corpus={CORPUS_NAME}" / f"split={target_split_name}" / f"language={LANGUAGE_CODE}"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    output_file = parquet_dir / "part-00000.parquet"
    
    # Remove existing file if present
    if output_file.exists():
        output_file.unlink()
    
    # Process in batches
    total_written = 0
    total_skipped = 0
    total_audio_size = 0
    rows = df.to_dict('records')
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Converting {target_split_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_rows = rows[start_idx:end_idx]
        
        # Process batch
        records, skipped, audio_size = process_batch(batch_rows, audio_dir, target_split_name)
        
        # Write batch
        if records:
            write_parquet_batch(records, output_file, append=(batch_idx > 0))
            total_written += len(records)
        
        total_skipped += skipped
        total_audio_size += audio_size
        
        # Clear memory
        del records
        gc.collect()
    
    total_hours = total_audio_size / SAMPLE_RATE / 3600
    print(f"✓ Saved {total_written} samples to {output_file}")
    print(f"  Skipped: {total_skipped}, Total audio: {total_hours:.2f}h")
    
    return {
        'split': target_split_name,
        'samples': total_written,
        'skipped': total_skipped,
        'hours': total_hours,
    }


def compute_stats(output_dir: str) -> str:
    """Compute dataset statistics TSV."""
    print(f"\n{'='*60}")
    print("Computing statistics...")
    print(f"{'='*60}")
    
    stats = []
    dataset_path = Path(output_dir)
    
    for parquet_file in dataset_path.rglob("*.parquet"):
        # Read only metadata columns to save memory
        table = pq.read_table(parquet_file, columns=['audio_size', 'corpus', 'split', 'language'])
        df = table.to_pandas()
        
        for (corpus, split, language), group in df.groupby(['corpus', 'split', 'language']):
            duration_hours = group['audio_size'].sum() / SAMPLE_RATE / 3600
            stats.append({
                'corpus': corpus,
                'split': split,
                'language': language,
                'num_samples': len(group),
                'duration_hours': round(duration_hours, 2),
            })
        
        del table, df
        gc.collect()
    
    stats_df = pd.DataFrame(stats)
    stats_path = dataset_path / "dataset_stats.tsv"
    stats_df.to_csv(stats_path, sep='\t', index=False)
    
    print(f"✓ Stats saved to {stats_path}")
    print(stats_df.to_string(index=False))
    
    return str(stats_path)


def main():
    parser = argparse.ArgumentParser(description="Convert RegSpeech12 to Parquet (Memory Efficient)")
    parser.add_argument("--output_dir", type=str, default="/root/thesis/data/regspeech12_parquet")
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT)
    parser.add_argument("--batch_size", type=int, default=200, 
                        help="Samples per batch. Reduce if OOM (try 100 or 50)")
    parser.add_argument("--split", type=str, default="all", 
                        choices=["train", "valid", "test", "all"],
                        help="Which split to process")
    
    args = parser.parse_args()
    
    print("RegSpeech12 → Parquet Conversion (Memory Efficient)")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    
    # Map source splits to target splits (valid → dev for omnilingual-asr)
    all_splits = [
        ('train', 'train', 'train.xlsx', 'train'),
        ('valid', 'dev', 'valid.xlsx', 'valid'),
        ('test', 'test', 'test.xlsx', 'test'),
    ]
    
    # Filter splits if specific one requested
    if args.split != "all":
        all_splits = [s for s in all_splits if s[0] == args.split]
    
    results = []
    for source_split, target_split, xlsx_name, audio_subdir in all_splits:
        result = process_split(
            xlsx_path=os.path.join(args.dataset_root, xlsx_name),
            audio_dir=os.path.join(args.dataset_root, audio_subdir),
            source_split_name=source_split,
            target_split_name=target_split,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        results.append(result)
        gc.collect()
    
    stats_path = compute_stats(args.output_dir)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    total_hours = 0
    for r in results:
        print(f"  {r['split']}: {r['samples']} samples, {r.get('hours', 0):.2f}h")
        total_hours += r.get('hours', 0)
    print(f"  Total: {total_hours:.2f}h")
    print(f"\nOutput: {args.output_dir}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()