#!/usr/bin/env python3
"""
RegSpeech12 Dataset Preparation Script for OmniLingual ASR Fine-tuning

Downloads the RegSpeech12 dataset from Kaggle and converts it to the
OmniLingual ASR manifest format (TSV + WRD files).

Dataset: https://www.kaggle.com/datasets/mdrezuwanhassan/regspeech12
- 100 hours of Bengali regional speech
- 80/10/10 train/val/test split
- 21,313 audio clips from 394+ speakers

Dataset Structure (from Kaggle):
    regspeech12/
    ├── train/           # ~17,049 audio files
    │   └── train_*.wav
    ├── valid/           # ~2,132 audio files
    │   └── valid_*.wav
    ├── test/            # ~2,132 audio files
    │   └── test_*.wav
    ├── train.xlsx       # Transcriptions for train split
    ├── valid.xlsx       # Transcriptions for valid split
    └── test.xlsx        # Transcriptions for test split

Usage:
    python prepare_regspeech12.py

Requirements:
    pip install kagglehub pandas openpyxl soundfile tqdm

Output Structure:
    dataset/regspeech12/
    ├── data_manifest/
    │   ├── train.tsv    # Training manifest
    │   ├── train.wrd    # Training transcriptions
    │   ├── valid.tsv    # Validation manifest
    │   ├── valid.wrd    # Validation transcriptions
    │   ├── test.tsv     # Test manifest
    │   └── test.wrd     # Test transcriptions
    └── wavs/            # All audio files (16kHz mono)
"""

import os
import sys
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

KAGGLE_DATASET = "mdrezuwanhassan/regspeech12"
OUTPUT_DIR = Path("dataset/regspeech12")
TARGET_SAMPLE_RATE = 16000  # OmniLingual expects 16kHz

# Split configuration: (split_name, audio_folder, xlsx_file)
SPLITS = [
    ("train", "train", "train.xlsx"),
    ("valid", "valid", "valid.xlsx"),
    ("test", "test", "test.xlsx"),
]

# ============================================================================
# Imports with helpful error messages
# ============================================================================

def check_dependencies():
    """Check and import required packages."""
    missing = []

    try:
        import kagglehub
    except ImportError:
        missing.append("kagglehub")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import openpyxl
    except ImportError:
        missing.append("openpyxl")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print("Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import kagglehub
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ============================================================================
# Helper Functions
# ============================================================================

def get_audio_info(audio_path: Path) -> tuple[int, int]:
    """
    Get audio sample rate and number of samples.

    Returns:
        (sample_rate, num_samples)
    """
    info = sf.info(audio_path)
    return info.samplerate, info.frames


def convert_to_16khz_mono(input_path: Path, output_path: Path) -> int:
    """
    Convert audio to 16kHz mono WAV format.

    Args:
        input_path: Source audio file
        output_path: Destination path

    Returns:
        Number of samples in the output file
    """
    # Read audio
    audio, sample_rate = sf.read(input_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        except ImportError:
            # Fallback: simple decimation (less accurate but works)
            if sample_rate % TARGET_SAMPLE_RATE == 0:
                factor = sample_rate // TARGET_SAMPLE_RATE
                audio = audio[::factor]
            else:
                print(f"Warning: Cannot resample {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz without librosa")
                print("  Install librosa: pip install librosa")

    # Write output
    sf.write(output_path, audio, TARGET_SAMPLE_RATE)

    return len(audio)


def find_xlsx_file(dataset_path: Path, xlsx_name: str) -> Optional[Path]:
    """Find an xlsx file in the dataset."""
    # Try direct path
    direct = dataset_path / xlsx_name
    if direct.exists():
        return direct

    # Search recursively
    for xlsx_file in dataset_path.rglob(xlsx_name):
        return xlsx_file

    # Try without extension match
    base_name = Path(xlsx_name).stem
    for xlsx_file in dataset_path.rglob(f"{base_name}.*"):
        if xlsx_file.suffix in [".xlsx", ".xls", ".csv"]:
            return xlsx_file

    return None


def find_audio_folder(dataset_path: Path, folder_name: str) -> Optional[Path]:
    """Find an audio folder in the dataset."""
    # Try direct path
    direct = dataset_path / folder_name
    if direct.exists() and direct.is_dir():
        return direct

    # Search recursively
    for folder in dataset_path.rglob(folder_name):
        if folder.is_dir():
            return folder

    return None


def load_xlsx_transcriptions(xlsx_path: Path) -> dict[str, str]:
    """
    Load transcriptions from Excel file.

    Returns:
        Dict mapping filename (with or without extension) to transcription
    """
    # Try reading with different engines
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(xlsx_path)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Find filename and transcription columns
    filename_cols = ["filename", "file", "audio", "path", "audio_path", "wav", "id", "name"]
    transcript_cols = ["transcription", "transcript", "text", "sentence", "label", "target"]

    filename_col = None
    transcript_col = None

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if filename_col is None:
            for fc in filename_cols:
                if fc in col_lower:
                    filename_col = col
                    break
        if transcript_col is None:
            for tc in transcript_cols:
                if tc in col_lower:
                    transcript_col = col
                    break

    # Fallback: use first and last columns
    if filename_col is None:
        filename_col = df.columns[0]
        print(f"  Warning: Using first column as filename: {filename_col}")

    if transcript_col is None:
        transcript_col = df.columns[-1]
        print(f"  Warning: Using last column as transcription: {transcript_col}")

    print(f"  Filename column: {filename_col}")
    print(f"  Transcription column: {transcript_col}")

    # Build mapping
    transcriptions = {}
    for _, row in df.iterrows():
        filename = str(row[filename_col]).strip()
        transcription = str(row[transcript_col]).strip()

        if pd.isna(row[transcript_col]) or transcription.lower() == "nan":
            continue

        # Store with and without extension
        transcriptions[filename] = transcription
        stem = Path(filename).stem
        if stem != filename:
            transcriptions[stem] = transcription

    return transcriptions


# ============================================================================
# Main Processing
# ============================================================================

def download_dataset() -> Path:
    """Download the dataset from Kaggle."""
    print("=" * 60)
    print("Step 1: Downloading RegSpeech12 from Kaggle")
    print("=" * 60)

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset downloaded to: {path}")

    return Path(path)


def explore_dataset(dataset_path: Path):
    """Explore and print the dataset structure."""
    print("\n" + "=" * 60)
    print("Step 2: Exploring dataset structure")
    print("=" * 60)

    print(f"\nContents of {dataset_path}:")

    def print_tree(path: Path, prefix: str = "  ", max_files: int = 5):
        items = sorted(path.iterdir())
        dirs = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]

        for d in dirs:
            file_count = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"{prefix}[DIR]  {d.name}/ ({file_count} files)")

        for i, f in enumerate(files):
            if i < max_files:
                print(f"{prefix}[FILE] {f.name}")
            elif i == max_files:
                print(f"{prefix}... and {len(files) - max_files} more files")
                break

    print_tree(dataset_path)


def process_split(
    dataset_path: Path,
    split_name: str,
    audio_folder_name: str,
    xlsx_name: str,
    wavs_dir: Path
) -> list[tuple[str, int, str]]:
    """
    Process a single split (train/valid/test).

    Returns:
        List of (filename, num_samples, transcription) tuples
    """
    print(f"\n--- Processing {split_name} split ---")

    # Find xlsx file
    xlsx_path = find_xlsx_file(dataset_path, xlsx_name)
    if xlsx_path is None:
        print(f"  ERROR: Could not find {xlsx_name}")
        return []

    print(f"  Found: {xlsx_path}")

    # Load transcriptions
    transcriptions = load_xlsx_transcriptions(xlsx_path)
    print(f"  Loaded {len(transcriptions)} transcriptions")

    # Find audio folder
    audio_folder = find_audio_folder(dataset_path, audio_folder_name)
    if audio_folder is None:
        print(f"  ERROR: Could not find {audio_folder_name}/ folder")
        return []

    print(f"  Audio folder: {audio_folder}")

    # Process audio files
    audio_files = list(audio_folder.glob("*.wav"))
    print(f"  Found {len(audio_files)} audio files")

    results = []
    skipped = 0

    for audio_path in tqdm(audio_files, desc=f"  {split_name}"):
        filename = audio_path.name
        stem = audio_path.stem

        # Find transcription
        transcription = transcriptions.get(filename) or transcriptions.get(stem)

        if transcription is None:
            skipped += 1
            continue

        # Output path
        out_path = wavs_dir / filename

        # Convert to 16kHz mono
        try:
            if out_path.exists():
                _, num_samples = get_audio_info(out_path)
            else:
                num_samples = convert_to_16khz_mono(audio_path, out_path)

            results.append((filename, num_samples, transcription))

        except Exception as e:
            print(f"\n  Error processing {filename}: {e}")
            skipped += 1

    print(f"  Processed: {len(results)}, Skipped: {skipped}")

    return results


def write_manifest(
    split_name: str,
    items: list[tuple[str, int, str]],
    manifest_dir: Path,
    wavs_path: str
):
    """Write TSV and WRD files for a split."""
    if not items:
        print(f"  Skipping {split_name} (no samples)")
        return

    tsv_path = manifest_dir / f"{split_name}.tsv"
    wrd_path = manifest_dir / f"{split_name}.wrd"

    # Write TSV
    with open(tsv_path, "w", encoding="utf-8") as f:
        # First line: path to wavs directory
        f.write(f"{wavs_path}\n")

        # Each subsequent line: filename<TAB>num_samples
        for filename, num_samples, _ in items:
            f.write(f"{filename}\t{num_samples}\n")

    # Write WRD
    with open(wrd_path, "w", encoding="utf-8") as f:
        for _, _, transcription in items:
            # Clean transcription (remove newlines, extra spaces)
            clean_text = " ".join(transcription.split())
            f.write(f"{clean_text}\n")

    print(f"  {split_name}: {len(items)} samples -> {tsv_path.name}, {wrd_path.name}")


def print_summary(split_counts: dict[str, int]):
    """Print final summary and next steps."""
    print("\n" + "=" * 60)
    print("DONE! Dataset prepared successfully")
    print("=" * 60)

    print(f"\nOutput location: {OUTPUT_DIR.absolute()}")

    print("\nDirectory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── data_manifest/")
    for split_name, count in split_counts.items():
        print(f"  │   ├── {split_name}.tsv  ({count} samples)")
        print(f"  │   ├── {split_name}.wrd")
    print(f"  └── wavs/")
    print(f"      └── *.wav")

    total = sum(split_counts.values())
    print(f"\nTotal samples: {total}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)

    print("\n1. IMPORTANT: Reinstall the package to register the asset card:")
    print("   pip install -e .")
    print("")
    print("   This registers the 'regspeech12' dataset name with its path.")
    print("   Asset card: src/omnilingual_asr/cards/datasets/regspeech12.yaml")

    print("\n2. Run training with the pre-configured config:")
    print("   python -m workflows.recipes.wav2vec2.asr \\")
    print("       --config-file workflows/recipes/wav2vec2/asr/configs/regspeech12-80hr.yaml \\")
    print("       output/")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("RegSpeech12 Dataset Preparation for OmniLingual ASR")
    print("=" * 60)

    # Step 1: Download
    dataset_path = download_dataset()

    # Step 2: Explore structure
    explore_dataset(dataset_path)

    # Step 3: Create output directories
    print("\n" + "=" * 60)
    print("Step 3: Creating output directories")
    print("=" * 60)

    wavs_dir = OUTPUT_DIR / "wavs"
    manifest_dir = OUTPUT_DIR / "data_manifest"

    wavs_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Wavs: {wavs_dir}")
    print(f"  Manifests: {manifest_dir}")

    # Step 4: Process each split
    print("\n" + "=" * 60)
    print("Step 4: Processing splits")
    print("=" * 60)

    split_results = {}
    for split_name, audio_folder, xlsx_file in SPLITS:
        items = process_split(
            dataset_path=dataset_path,
            split_name=split_name,
            audio_folder_name=audio_folder,
            xlsx_name=xlsx_file,
            wavs_dir=wavs_dir
        )
        split_results[split_name] = items

    # Step 5: Write manifests
    print("\n" + "=" * 60)
    print("Step 5: Writing manifest files")
    print("=" * 60)

    wavs_path = str(OUTPUT_DIR / "wavs")
    split_counts = {}

    for split_name, items in split_results.items():
        write_manifest(split_name, items, manifest_dir, wavs_path)
        split_counts[split_name] = len(items)

    # Summary
    print_summary(split_counts)


if __name__ == "__main__":
    main()
