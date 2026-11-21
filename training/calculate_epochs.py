#!/usr/bin/env python3
"""
Calculate training epochs for a fine-tuning run.

This script calculates the number of epochs using official fairseq2 methods:
1. Checkpoint (OFFICIAL) - reads _data_epoch_nr directly from trainer checkpoint
2. Training logs (OFFICIAL) - parses "End of epoch X" and "Data Epoch: X" from logs
3. Config + manifest (ESTIMATED) - calculates from steps and batch size

Usage:
    # From checkpoint (most reliable - official fairseq2 state)
    python calculate_epochs.py --checkpoint /path/to/checkpoints/step_10000

    # From training logs
    python calculate_epochs.py --log /path/to/logs/rank_0.log

    # From config and manifest (estimated)
    python calculate_epochs.py --config /path/to/config.yaml --manifest /path/to/train.tsv

    # Full training output directory (auto-detects all files)
    python calculate_epochs.py --output-dir /path/to/output/ws_1.xxxxx

    # Manual calculation
    python calculate_epochs.py --steps 10000 --samples 17049 --batch-size 8 --grad-accum 8 --world-size 1
"""

import argparse
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def parse_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Read epoch information directly from fairseq2 trainer checkpoint.

    This is the OFFICIAL way to get epoch info - reads _data_epoch_nr
    which is the epoch counter maintained by fairseq2's Trainer class.

    Reference: fairseq2/recipe/trainer.py line 316, 538, 1152
    """
    import torch
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)

    # Handle both checkpoint dir and trainer file
    if checkpoint_path.is_dir():
        trainer_file = checkpoint_path / "trainer" / "rank_00.pt"
        if not trainer_file.exists():
            trainer_file = checkpoint_path / "trainer" / "rank_0.pt"
    else:
        trainer_file = checkpoint_path

    if not trainer_file.exists():
        raise FileNotFoundError(f"Trainer checkpoint not found: {trainer_file}")

    ckpt = torch.load(trainer_file, map_location='cpu', weights_only=False)

    return {
        "step_nr": ckpt.get("_step_nr", 0),
        "data_epoch_nr": ckpt.get("_data_epoch_nr", 0),
        "base_wall_time": ckpt.get("_base_wall_time", 0),
    }


def parse_log_file(log_path: str) -> Dict[str, Any]:
    """Parse fairseq2 training log to extract training statistics."""
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    stats = {
        "start_time": None,
        "end_time": None,
        "final_step": 0,
        "final_data_epoch": 0,
        "total_examples": 0,
        "total_time_seconds": 0,
        "final_wer": None,
        "final_loss": None,
        "epoch_boundaries": [],  # List of (epoch, step) when each epoch ended
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Get start time from first line
    if lines:
        match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', lines[0])
        if match:
            stats["start_time"] = match.group()

    # Parse epoch boundaries (official fairseq2 log format)
    seen_epochs = set()
    for line in lines:
        if "End of epoch" in line:
            # Format: "End of epoch X reached after step Y."
            epoch_match = re.search(r'End of epoch (\d+) reached after step (\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                step = int(epoch_match.group(2))
                # Deduplicate (training may have been resumed)
                if epoch not in seen_epochs:
                    stats["epoch_boundaries"].append((epoch, step))
                    seen_epochs.add(epoch)

    # Parse from end of file for final stats
    for line in reversed(lines):
        # Look for final training metrics
        if "Train Metrics" in line and stats["final_step"] == 0:
            step_match = re.search(r'step (\d+)', line)
            epoch_match = re.search(r'Data Epoch: (\d+)', line)
            examples_match = re.search(r'Total Number of Examples: ([\d,]+)', line)
            loss_match = re.search(r'CTC Loss: ([\d.]+)', line)

            if step_match:
                stats["final_step"] = int(step_match.group(1))
            if epoch_match:
                stats["final_data_epoch"] = int(epoch_match.group(1))
            if examples_match:
                stats["total_examples"] = int(examples_match.group(1).replace(',', ''))
            if loss_match:
                stats["final_loss"] = float(loss_match.group(1))

        # Look for validation WER
        if "Word Error Rate (WER)" in line and stats["final_wer"] is None:
            wer_match = re.search(r'Word Error Rate \(WER\): ([\d.]+)', line)
            if wer_match:
                stats["final_wer"] = float(wer_match.group(1))

        # Look for task finished line
        if "Task finished in" in line:
            # Handle numbers with commas like "23,036"
            time_match = re.search(r'([\d,]+) second', line)
            if time_match:
                stats["total_time_seconds"] = int(time_match.group(1).replace(',', ''))
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
            if match:
                stats["end_time"] = match.group()

    return stats


def count_manifest_samples(manifest_path: str) -> int:
    """Count samples in a TSV manifest file (excludes header line)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    # First line is typically the root path, rest are samples
    return len(lines) - 1


def parse_config(config_path: str) -> Dict[str, Any]:
    """Parse training config.yaml file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return {
        "num_steps": config.get("regime", {}).get("num_steps", 0),
        "batch_size": config.get("dataset", {}).get("asr_task_config", {}).get("batch_size", 8),
        "grad_accum": config.get("trainer", {}).get("grad_accumulation", {}).get("num_batches", 1),
        "dataset_name": config.get("dataset", {}).get("name", "unknown"),
        "model_name": config.get("model", {}).get("name", "unknown"),
        "learning_rate": config.get("optimizer", {}).get("config", {}).get("lr", 0),
    }


def calculate_epochs_from_samples(
    total_samples: int,
    num_steps: int,
    batch_size: int,
    grad_accum: int,
    world_size: int = 1
) -> float:
    """
    Calculate approximate epochs from training parameters.

    Note: This is an approximation. Actual epochs may differ due to:
    - Dynamic batching (LENGTH strategy)
    - Dropped samples (too long/short)
    - Padding and batching variations
    """
    effective_batch_size = batch_size * grad_accum * world_size
    total_samples_seen = num_steps * effective_batch_size
    epochs = total_samples_seen / total_samples
    return epochs


def calculate_epochs_from_examples(total_examples: int, training_samples: int) -> float:
    """Calculate exact epochs from total examples processed."""
    return total_examples / training_samples


def format_time(seconds: int) -> str:
    """Format seconds into human-readable time."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate training epochs for fine-tuning runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    parser.add_argument("--output-dir", type=str, help="Training output directory (auto-detects files)")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory (e.g., step_10000)")
    parser.add_argument("--log", type=str, help="Path to rank_0.log file")
    parser.add_argument("--config", type=str, help="Path to config.yaml file")
    parser.add_argument("--manifest", type=str, help="Path to train.tsv manifest file")

    # Manual calculation options
    parser.add_argument("--steps", type=int, help="Number of training steps")
    parser.add_argument("--samples", type=int, help="Number of training samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs (default: 1)")

    args = parser.parse_args()

    # Auto-detect files from output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not args.log:
            log_path = output_dir / "logs" / "rank_0.log"
            if log_path.exists():
                args.log = str(log_path)
        if not args.config:
            config_path = output_dir / "config.yaml"
            if config_path.exists():
                args.config = str(config_path)
        if not args.checkpoint:
            # Find latest checkpoint
            checkpoints_dir = output_dir / "checkpoints"
            if checkpoints_dir.exists():
                step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
                if step_dirs:
                    latest = max(step_dirs, key=lambda d: int(d.name.split("_")[1]))
                    args.checkpoint = str(latest)

    print("=" * 60)
    print("TRAINING EPOCHS CALCULATOR")
    print("=" * 60)

    training_samples = None
    config_info = None
    log_stats = None
    checkpoint_info = None

    # Method 0: Parse checkpoint if available (OFFICIAL - most reliable)
    if args.checkpoint:
        try:
            checkpoint_info = parse_checkpoint(args.checkpoint)
            print(f"\n[Checkpoint] {args.checkpoint} (OFFICIAL)")
            print(f"  Step: {checkpoint_info['step_nr']}")
            print(f"  Data Epoch (fairseq2 _data_epoch_nr): {checkpoint_info['data_epoch_nr']}")
            print(f"  Wall time: {format_time(int(checkpoint_info['base_wall_time']))}")
        except Exception as e:
            print(f"[Warning] Could not parse checkpoint: {e}")

    # Parse config if available
    if args.config:
        try:
            config_info = parse_config(args.config)
            print(f"\n[Config] {args.config}")
            print(f"  Model: {config_info['model_name']}")
            print(f"  Dataset: {config_info['dataset_name']}")
            print(f"  Steps: {config_info['num_steps']}")
            print(f"  Batch size: {config_info['batch_size']}")
            print(f"  Grad accumulation: {config_info['grad_accum']}")
            print(f"  Learning rate: {config_info['learning_rate']}")
        except Exception as e:
            print(f"[Warning] Could not parse config: {e}")

    # Count manifest samples if available
    if args.manifest:
        try:
            training_samples = count_manifest_samples(args.manifest)
            print(f"\n[Manifest] {args.manifest}")
            print(f"  Training samples: {training_samples:,}")
        except Exception as e:
            print(f"[Warning] Could not parse manifest: {e}")

    # Use manual samples if provided
    if args.samples:
        training_samples = args.samples
        print(f"\n[Manual] Training samples: {training_samples:,}")

    # Parse log file if available (most accurate method)
    if args.log:
        try:
            log_stats = parse_log_file(args.log)
            print(f"\n[Log] {args.log}")
            print(f"  Start: {log_stats['start_time']}")
            print(f"  End: {log_stats['end_time']}")
            print(f"  Duration: {format_time(log_stats['total_time_seconds'])}")
            print(f"  Final step: {log_stats['final_step']}")
            print(f"  Data epoch (from log): {log_stats['final_data_epoch']}")
            print(f"  Total examples processed: {log_stats['total_examples']:,}")
            if log_stats['final_wer']:
                print(f"  Final WER: {log_stats['final_wer']:.2f}%")
            if log_stats['final_loss']:
                print(f"  Final CTC Loss: {log_stats['final_loss']:.3f}")
            if log_stats['epoch_boundaries']:
                print(f"  Epoch boundaries (official 'End of epoch X reached after step Y'):")
                for epoch, step in log_stats['epoch_boundaries']:
                    print(f"    - Epoch {epoch} completed at step {step}")
        except Exception as e:
            print(f"[Warning] Could not parse log: {e}")

    # Calculate epochs
    print("\n" + "=" * 60)
    print("EPOCH CALCULATIONS")
    print("=" * 60)

    # Method 0: From checkpoint (OFFICIAL - fairseq2 internal state)
    if checkpoint_info:
        data_epoch_nr = checkpoint_info['data_epoch_nr']
        completed_epochs = data_epoch_nr - 1  # _data_epoch_nr is 1-indexed, current epoch
        print(f"\n[Method 0] From checkpoint (OFFICIAL - fairseq2 _data_epoch_nr)")
        print(f"  _data_epoch_nr from checkpoint: {data_epoch_nr}")
        print(f"  This means training was IN epoch {data_epoch_nr} when stopped")
        print(f"  >>> COMPLETED EPOCHS: {completed_epochs}")

        # If we have log info, calculate partial epoch progress
        if log_stats and log_stats.get('epoch_boundaries'):
            boundaries = log_stats['epoch_boundaries']
            last_boundary = boundaries[-1] if boundaries else None
            if last_boundary:
                last_epoch, last_step = last_boundary
                final_step = checkpoint_info['step_nr']
                if final_step > last_step:
                    # Estimate steps per epoch from boundaries
                    if len(boundaries) >= 2:
                        steps_per_epoch = boundaries[-1][1] - boundaries[-2][1]
                    else:
                        steps_per_epoch = last_step  # First epoch
                    partial = (final_step - last_step) / steps_per_epoch
                    total_epochs = completed_epochs + partial
                    print(f"  Partial epoch {data_epoch_nr} progress: {partial:.2f}")
                    print(f"  >>> TOTAL EPOCHS: {total_epochs:.2f}")

    # Method 1: From log file
    if log_stats and training_samples:
        exact_epochs = calculate_epochs_from_examples(
            log_stats['total_examples'],
            training_samples
        )
        print(f"\n[Method 1] From log (RECOMMENDED - most accurate)")
        print(f"  Total examples: {log_stats['total_examples']:,}")
        print(f"  Training samples: {training_samples:,}")
        print(f"  >>> EPOCHS: {exact_epochs:.2f}")

    # Method 2: From config (estimated)
    if config_info and training_samples:
        num_steps = args.steps or config_info['num_steps']
        batch_size = args.batch_size if args.batch_size != 8 else config_info['batch_size']
        grad_accum = args.grad_accum if args.grad_accum != 1 else config_info['grad_accum']
        world_size = args.world_size

        estimated_epochs = calculate_epochs_from_samples(
            training_samples, num_steps, batch_size, grad_accum, world_size
        )
        print(f"\n[Method 2] From config (estimated)")
        print(f"  Steps: {num_steps}")
        print(f"  Effective batch size: {batch_size} × {grad_accum} × {world_size} = {batch_size * grad_accum * world_size}")
        print(f"  Training samples: {training_samples:,}")
        print(f"  >>> EPOCHS (estimated): {estimated_epochs:.2f}")
        print(f"  Note: May differ from actual due to dynamic batching")

    # Method 3: Manual calculation
    if args.steps and args.samples:
        manual_epochs = calculate_epochs_from_samples(
            args.samples, args.steps, args.batch_size, args.grad_accum, args.world_size
        )
        print(f"\n[Method 3] Manual calculation")
        print(f"  Steps: {args.steps}")
        print(f"  Effective batch size: {args.batch_size} × {args.grad_accum} × {args.world_size} = {args.batch_size * args.grad_accum * args.world_size}")
        print(f"  Training samples: {args.samples:,}")
        print(f"  >>> EPOCHS: {manual_epochs:.2f}")

    # Summary for thesis
    print("\n" + "=" * 60)
    print("SUMMARY FOR THESIS")
    print("=" * 60)

    # Determine best epoch value
    best_epochs = None
    epoch_source = None

    if checkpoint_info:
        # Official method: checkpoint _data_epoch_nr
        completed = checkpoint_info['data_epoch_nr'] - 1
        # Try to get partial from log
        if log_stats and log_stats.get('epoch_boundaries'):
            boundaries = log_stats['epoch_boundaries']
            if boundaries:
                last_epoch, last_step = boundaries[-1]
                final_step = checkpoint_info['step_nr']
                if final_step > last_step and len(boundaries) >= 2:
                    steps_per_epoch = boundaries[-1][1] - boundaries[-2][1]
                    partial = (final_step - last_step) / steps_per_epoch
                    best_epochs = completed + partial
                else:
                    best_epochs = float(completed)
            else:
                best_epochs = float(completed)
        else:
            best_epochs = float(completed)
        epoch_source = "checkpoint (_data_epoch_nr)"
    elif log_stats and training_samples:
        best_epochs = calculate_epochs_from_examples(log_stats['total_examples'], training_samples)
        epoch_source = "log (total_examples / training_samples)"

    if best_epochs is not None:
        step_nr = checkpoint_info['step_nr'] if checkpoint_info else (log_stats['final_step'] if log_stats else 0)
        wall_time = checkpoint_info['base_wall_time'] if checkpoint_info else (log_stats['total_time_seconds'] if log_stats else 0)

        print(f"""
| Parameter | Value |
|-----------|-------|
| Training samples | {training_samples:,} |
| Training steps | {step_nr:,} |""")
        if log_stats:
            print(f"| Total examples processed | {log_stats['total_examples']:,} |")
        print(f"| **Epochs** | **{best_epochs:.1f}** (from {epoch_source}) |")
        print(f"| Training time | {format_time(int(wall_time))} |")
        if config_info:
            print(f"| Learning rate | {config_info['learning_rate']} |")
        if log_stats and log_stats['final_wer']:
            print(f"| Final WER | {log_stats['final_wer']:.2f}% |")


if __name__ == "__main__":
    main()
