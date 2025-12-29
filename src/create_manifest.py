"""
RegSpeech12 → Manifest Conversion for Omnilingual ASR Fine-tuning

This script creates manifest files (TSV/JSON) for the ManifestAsrDataset backend,
which is simpler than Parquet for single-corpus fine-tuning.

Manifest format (TSV):
    audio_path\ttext\tduration\tlanguage

Usage:
    python create_manifest.py --output_dir /root/thesis/data/regspeech12_manifest
    python create_manifest.py --output_dir /root/thesis/data/regspeech12_manifest --format jsonl
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import soundfile as sf
from tqdm import tqdm


# Constants
DATASET_ROOT = "/root/.cache/kagglehub/datasets/mdrezuwanhassan/regspeech12/versions/1"
SAMPLE_RATE = 16000
LANGUAGE_CODE = "ben_Beng"


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        # Fallback: read the file
        audio, sr = sf.read(audio_path)
        return len(audio) / sr


def extract_dialect(filename: str) -> str:
    """Extract dialect from filename like 'train_barishal_0001.wav'."""
    parts = filename.replace('.wav', '').split('_')
    return parts[1] if len(parts) >= 2 else 'unknown'


def create_manifest_tsv(
    xlsx_path: str,
    audio_dir: str,
    output_path: str,
    split_name: str,
) -> dict:
    """Create TSV manifest file."""
    
    print(f"\nProcessing {split_name}...")
    df = pd.read_excel(xlsx_path)
    print(f"  Total samples: {len(df)}")
    
    records = []
    skipped = 0
    total_duration = 0.0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating {split_name} manifest"):
        audio_path = os.path.join(audio_dir, row['file_name'])
        
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        try:
            duration = get_audio_duration(audio_path)
            total_duration += duration
            
            records.append({
                'audio_path': audio_path,
                'text': str(row['transcripts']).strip(),
                'duration': round(duration, 3),
                'language': LANGUAGE_CODE,
                'dialect': extract_dialect(row['file_name']),
                'file_name': row['file_name'],
            })
        except Exception as e:
            print(f"  Error processing {row['file_name']}: {e}")
            skipped += 1
    
    # Write TSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("audio_path\ttext\tduration\tlanguage\tdialect\tfile_name\n")
        # Data
        for r in records:
            f.write(f"{r['audio_path']}\t{r['text']}\t{r['duration']}\t{r['language']}\t{r['dialect']}\t{r['file_name']}\n")
    
    print(f"  ✓ Saved {len(records)} samples to {output_path}")
    print(f"  Skipped: {skipped}")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    
    return {
        'split': split_name,
        'samples': len(records),
        'skipped': skipped,
        'hours': total_duration / 3600,
        'output_path': output_path,
    }


def create_manifest_jsonl(
    xlsx_path: str,
    audio_dir: str,
    output_path: str,
    split_name: str,
) -> dict:
    """Create JSON Lines manifest file."""
    
    print(f"\nProcessing {split_name}...")
    df = pd.read_excel(xlsx_path)
    print(f"  Total samples: {len(df)}")
    
    records = []
    skipped = 0
    total_duration = 0.0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating {split_name} manifest"):
        audio_path = os.path.join(audio_dir, row['file_name'])
        
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        try:
            duration = get_audio_duration(audio_path)
            total_duration += duration
            
            records.append({
                'audio_path': audio_path,
                'text': str(row['transcripts']).strip(),
                'duration': round(duration, 3),
                'n_frames': int(duration * SAMPLE_RATE),  # Audio samples count
                'language': LANGUAGE_CODE,
                'dialect': extract_dialect(row['file_name']),
            })
        except Exception as e:
            print(f"  Error processing {row['file_name']}: {e}")
            skipped += 1
    
    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Saved {len(records)} samples to {output_path}")
    print(f"  Skipped: {skipped}")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    
    return {
        'split': split_name,
        'samples': len(records),
        'skipped': skipped,
        'hours': total_duration / 3600,
        'output_path': output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Create manifest files for RegSpeech12")
    parser.add_argument("--output_dir", type=str, default="/root/thesis/data/regspeech12_manifest",
                        help="Output directory for manifest files")
    parser.add_argument("--format", type=str, default="tsv", choices=["tsv", "jsonl"],
                        help="Manifest format")
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT,
                        help="Root directory of RegSpeech12")
    
    args = parser.parse_args()
    
    print("="*60)
    print("RegSpeech12 Manifest Creation")
    print("="*60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Format: {args.format}")
    
    # Splits configuration
    splits = {
        'train': {
            'xlsx': os.path.join(args.dataset_root, 'train.xlsx'),
            'audio_dir': os.path.join(args.dataset_root, 'train'),
        },
        'valid': {
            'xlsx': os.path.join(args.dataset_root, 'valid.xlsx'),
            'audio_dir': os.path.join(args.dataset_root, 'valid'),
        },
        'test': {
            'xlsx': os.path.join(args.dataset_root, 'test.xlsx'),
            'audio_dir': os.path.join(args.dataset_root, 'test'),
        },
    }
    
    # Create function based on format
    create_fn = create_manifest_jsonl if args.format == "jsonl" else create_manifest_tsv
    ext = "jsonl" if args.format == "jsonl" else "tsv"
    
    # Process each split
    results = []
    for split_name, config in splits.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.{ext}")
        result = create_fn(
            xlsx_path=config['xlsx'],
            audio_dir=config['audio_dir'],
            output_path=output_path,
            split_name=split_name,
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("MANIFEST CREATION COMPLETE")
    print("="*60)
    total_hours = 0
    for r in results:
        print(f"  {r['split']}: {r['samples']} samples, {r['hours']:.2f}h")
        total_hours += r['hours']
    print(f"  Total: {total_hours:.2f} hours")
    print(f"\nOutput directory: {args.output_dir}")
    
    # Create a simple README
    readme_path = os.path.join(args.output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"RegSpeech12 Manifest Files\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Format: {args.format}\n\n")
        for r in results:
            f.write(f"{r['split']}: {r['samples']} samples, {r['hours']:.2f} hours\n")


if __name__ == "__main__":
    main()
