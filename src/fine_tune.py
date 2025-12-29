"""
Fine-Tuning Script for Omnilingual ASR on Bengali Regional Dialects
Dataset: RegSpeech12 (Train: ~80h, Valid: ~10h)
Model: omniASR_LLM_1B_v2

Usage:
    python finetune.py --config fine_tune_config.yaml
    python finetune.py --epochs 10 --batch_size 4 --lr 1e-5
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

import pandas as pd
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
import yaml

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional: TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""
    
    # Paths
    dataset_root: str = "/root/.cache/kagglehub/datasets/mdrezuwanhassan/regspeech12/versions/1"
    output_dir: str = "/root/thesis/checkpoints"
    
    # Model
    model_card: str = "omniASR_LLM_1B_v2"
    freeze_encoder: bool = False  # Optionally freeze encoder layers
    freeze_encoder_layers: int = 0  # Number of encoder layers to freeze (0 = none)
    
    # Data
    train_xlsx: str = "train.xlsx"
    valid_xlsx: str = "valid.xlsx"
    audio_dir_train: str = "train"
    audio_dir_valid: str = "valid"
    dialect_filter: Optional[str] = None  # None = all dialects, or specific like "barishal"
    max_audio_length_sec: float = 30.0  # Skip audio longer than this
    
    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * grad_accum
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_ratio: float = 0.1  # 10% of total steps for warmup
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False  # Use bf16 instead of fp16 (better for newer GPUs)
    
    # Validation & checkpointing
    eval_steps: int = 500  # Evaluate every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    save_total_limit: int = 3  # Keep only last N checkpoints
    early_stopping_patience: int = 5  # Stop if no improvement for N evals
    metric_for_best_model: str = "wer"  # "wer" or "cer"
    
    # Logging
    logging_steps: int = 50
    log_to_wandb: bool = False
    wandb_project: str = "bengali-asr-finetune"
    wandb_run_name: Optional[str] = None
    log_to_tensorboard: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


def load_config(config_path: Optional[str] = None, **overrides) -> TrainingConfig:
    """Load config from YAML file and apply overrides."""
    config_dict = {}
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    
    # Apply command line overrides
    config_dict.update({k: v for k, v in overrides.items() if v is not None})
    
    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, path: str):
    """Save config to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


# =============================================================================
# Dataset
# =============================================================================

class RegSpeech12Dataset(Dataset):
    """Dataset for RegSpeech12 Bengali dialect speech."""
    
    def __init__(
        self,
        xlsx_path: str,
        audio_dir: str,
        dialect_filter: Optional[str] = None,
        max_audio_length_sec: float = 30.0,
        processor = None,
    ):
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_audio_length_sec = max_audio_length_sec
        
        # Load metadata
        df = pd.read_excel(xlsx_path)
        
        # Filter by dialect if specified
        if dialect_filter:
            if dialect_filter != "all":
                prefix = f"train_{dialect_filter}_" if "train" in xlsx_path else f"valid_{dialect_filter}_"
                df = df[df['file_name'].str.startswith(prefix)]
        
        # Filter out missing files and build sample list
        self.samples = []
        for _, row in df.iterrows():
            audio_path = os.path.join(audio_dir, row['file_name'])
            if os.path.exists(audio_path):
                self.samples.append({
                    'file_name': row['file_name'],
                    'audio_path': audio_path,
                    'transcript': row['transcripts'],
                })
        
        logging.info(f"Loaded {len(self.samples)} samples from {xlsx_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio_path': sample['audio_path'],
            'transcript': sample['transcript'],
            'file_name': sample['file_name'],
        }


class DataCollator:
    """
    Collate function for batching audio samples.
    Handles audio loading and preprocessing.
    """
    
    def __init__(self, processor, max_audio_length_sec: float = 30.0):
        self.processor = processor
        self.max_audio_length_sec = max_audio_length_sec
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        audio_paths = [item['audio_path'] for item in batch]
        transcripts = [item['transcript'] for item in batch]
        file_names = [item['file_name'] for item in batch]
        
        # Process audio through the model's processor
        # This will depend on the specific model implementation
        processed = self.processor.prepare_batch(
            audio_paths=audio_paths,
            transcripts=transcripts,
            lang=["ben_Beng"] * len(audio_paths),
        )
        
        processed['file_names'] = file_names
        processed['raw_transcripts'] = transcripts
        
        return processed


# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                   float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then cosine decay."""
    import math
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_scheduler(scheduler_type: str, optimizer, num_warmup_steps: int, num_training_steps: int):
    """Get learning rate scheduler."""
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class CheckpointManager:
    """Manage model checkpoints with limit on total saved."""
    
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.checkpoints = []
    
    def save(self, model, optimizer, scheduler, scaler, step: int, epoch: int, 
             metrics: Dict[str, float], config: TrainingConfig, is_best: bool = False):
        """Save checkpoint and manage limit."""
        
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save optimizer, scheduler, scaler
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'step': step,
            'epoch': epoch,
            'metrics': metrics,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save config
        save_config(config, os.path.join(checkpoint_dir, "fine_tune_config.yaml"))
        
        # Save metrics
        with open(os.path.join(checkpoint_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.checkpoints.append(checkpoint_dir)
        
        # Also save as best if applicable
        if is_best:
            best_dir = os.path.join(self.output_dir, "best_model")
            if os.path.exists(best_dir):
                import shutil
                shutil.rmtree(best_dir)
            import shutil
            shutil.copytree(checkpoint_dir, best_dir)
            logging.info(f"New best model saved to {best_dir}")
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint) and "best_model" not in old_checkpoint:
                import shutil
                shutil.rmtree(old_checkpoint)
                logging.info(f"Removed old checkpoint: {old_checkpoint}")
        
        logging.info(f"Saved checkpoint to {checkpoint_dir}")
        return checkpoint_dir
    
    def load_latest(self, model, optimizer, scheduler, scaler):
        """Load the latest checkpoint."""
        checkpoints = [d for d in os.listdir(self.output_dir) 
                       if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.output_dir, d))]
        
        if not checkpoints:
            return None, 0, 0
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest = os.path.join(self.output_dir, checkpoints[-1])
        
        return self.load(latest, model, optimizer, scheduler, scaler)
    
    def load(self, checkpoint_dir: str, model, optimizer, scheduler, scaler):
        """Load a specific checkpoint."""
        logging.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load model
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        
        # Load training state
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"))
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if scaler and state['scaler']:
            scaler.load_state_dict(state['scaler'])
        
        return state['metrics'], state['step'], state['epoch']


# =============================================================================
# Trainer
# =============================================================================

class ASRTrainer:
    """Trainer for fine-tuning ASR model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self._setup_logging()
        
        # Set seed
        set_seed(config.seed)
        
        # Initialize components
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.checkpoint_manager = CheckpointManager(config.output_dir, config.save_total_limit)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.writer = None
        if config.log_to_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = os.path.join(config.output_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.writer = SummaryWriter(log_dir)
        
        if config.log_to_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=asdict(config),
            )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.config.output_dir, "training.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup(self):
        """Initialize model, data, optimizer, etc."""
        logging.info("Setting up trainer...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Config: {asdict(self.config)}")
        
        # Load model and processor
        self._load_model()
        
        # Setup data
        self._setup_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup mixed precision
        if self.config.fp16 or self.config.bf16:
            self.scaler = GradScaler()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._resume_from_checkpoint()
        
        logging.info("Setup complete!")

    def _load_model(self):
        """Load pretrained OmniASR model for fine-tuning."""
        logging.info(f"Loading model: {self.config.model_card}")

        import torch
        from omnilingual_asr.models.wav2vec2_llama.factory import create_model

        # --------------------------------------------------
        # Create model + processor via factory
        # --------------------------------------------------
        model, processor = create_model(
            model_card=self.config.model_card,
            device=self.device,
            dtype=torch.float16 if self.config.fp16 else torch.float32,
        )

        self.model = model
        self.processor = processor

        # --------------------------------------------------
        # Optional: freeze encoder layers
        # --------------------------------------------------
        if self.config.freeze_encoder:
            self._freeze_encoder_layers()

        # Enable training mode
        self.model.train()

        # --------------------------------------------------
        # Log model info
        # --------------------------------------------------
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

    def _freeze_encoder_layers(self):
        """Freeze encoder layers for transfer learning."""
        logging.info(f"Freezing encoder layers...")
        
        # This is model-specific - adjust based on actual architecture
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if "encoder" in name.lower():
                if self.config.freeze_encoder_layers > 0:
                    # Freeze specific number of layers
                    layer_num = self._extract_layer_num(name)
                    if layer_num is not None and layer_num < self.config.freeze_encoder_layers:
                        param.requires_grad = False
                        frozen_count += 1
                else:
                    # Freeze all encoder layers
                    param.requires_grad = False
                    frozen_count += 1
        
        logging.info(f"Frozen {frozen_count} encoder parameters")
    
    def _extract_layer_num(self, name: str) -> Optional[int]:
        """Extract layer number from parameter name."""
        import re
        match = re.search(r'layer[._]?(\d+)', name.lower())
        if match:
            return int(match.group(1))
        return None
    
    def _setup_data(self):
        """Setup data loaders."""
        logging.info("Setting up data loaders...")
        
        train_xlsx = os.path.join(self.config.dataset_root, self.config.train_xlsx)
        valid_xlsx = os.path.join(self.config.dataset_root, self.config.valid_xlsx)
        train_audio_dir = os.path.join(self.config.dataset_root, self.config.audio_dir_train)
        valid_audio_dir = os.path.join(self.config.dataset_root, self.config.audio_dir_valid)
        
        # Create datasets
        train_dataset = RegSpeech12Dataset(
            xlsx_path=train_xlsx,
            audio_dir=train_audio_dir,
            dialect_filter=self.config.dialect_filter,
            max_audio_length_sec=self.config.max_audio_length_sec,
            processor=self.processor,
        )
        
        valid_dataset = RegSpeech12Dataset(
            xlsx_path=valid_xlsx,
            audio_dir=valid_audio_dir,
            dialect_filter=self.config.dialect_filter,
            max_audio_length_sec=self.config.max_audio_length_sec,
            processor=self.processor,
        )
        
        # Create data collator
        collator = DataCollator(
            processor=self.processor,
            max_audio_length_sec=self.config.max_audio_length_sec,
        )
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )
        
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
        )
        
        logging.info(f"Train samples: {len(train_dataset)}")
        logging.info(f"Valid samples: {len(valid_dataset)}")
        logging.info(f"Train batches: {len(self.train_dataloader)}")
        logging.info(f"Valid batches: {len(self.valid_dataloader)}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        logging.info("Setting up optimizer and scheduler...")
        
        # Filter parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        total_training_steps = num_update_steps_per_epoch * self.config.epochs
        num_warmup_steps = int(total_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps,
            total_training_steps,
        )
        
        logging.info(f"Total training steps: {total_training_steps}")
        logging.info(f"Warmup steps: {num_warmup_steps}")
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint."""
        checkpoint_path = self.config.resume_from_checkpoint
        
        if checkpoint_path == "latest":
            metrics, step, epoch = self.checkpoint_manager.load_latest(
                self.model, self.optimizer, self.scheduler, self.scaler
            )
        else:
            metrics, step, epoch = self.checkpoint_manager.load(
                checkpoint_path, self.model, self.optimizer, self.scheduler, self.scaler
            )
        
        if metrics:
            self.global_step = step
            self.current_epoch = epoch
            self.best_metric = metrics.get(self.config.metric_for_best_model, float('inf'))
            logging.info(f"Resumed from step {step}, epoch {epoch}")
    
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            logging.info(f"\n{'='*60}")
            logging.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logging.info(f"{'='*60}")
            
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Log epoch metrics
            logging.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Check for early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Final evaluation
        logging.info("\nFinal evaluation...")
        final_metrics = self.evaluate()
        logging.info(f"Final metrics: {final_metrics}")
        
        # Cleanup
        if self.writer:
            self.writer.close()
        if self.config.log_to_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        logging.info("Training complete!")
        return final_metrics
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=True,
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Forward pass
            loss = self._training_step(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step,
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics({
                        'train/loss': avg_loss,
                        'train/learning_rate': current_lr,
                        'train/epoch': self.current_epoch,
                    }, self.global_step)
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics({
                        'eval/wer': eval_metrics['wer'],
                        'eval/cer': eval_metrics['cer'],
                        'eval/loss': eval_metrics.get('loss', 0),
                    }, self.global_step)
                    
                    # Check for best model
                    current_metric = eval_metrics[self.config.metric_for_best_model]
                    is_best = current_metric < self.best_metric
                    
                    if is_best:
                        self.best_metric = current_metric
                        self.patience_counter = 0
                        logging.info(f"New best {self.config.metric_for_best_model}: {current_metric:.4f}")
                    else:
                        self.patience_counter += 1
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.checkpoint_manager.save(
                            self.model, self.optimizer, self.scheduler, self.scaler,
                            self.global_step, self.current_epoch, eval_metrics,
                            self.config, is_best=is_best
                        )
                    
                    # Return to training mode
                    self.model.train()
        
        return {'loss': total_loss / num_batches}
    
    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step."""
        # Move batch to device
        batch = self._move_to_device(batch)
        
        # Forward pass with mixed precision
        dtype = torch.float16 if self.config.fp16 else (torch.bfloat16 if self.config.bf16 else torch.float32)
        
        with autocast(enabled=self.config.fp16 or self.config.bf16, dtype=dtype):
            # This is model-specific - adjust based on actual API
            # The model should return a loss when given inputs and labels
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        return loss
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        logging.info("Running evaluation...")
        self.model.eval()
        
        all_references = []
        all_hypotheses = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.valid_dataloader, desc="Evaluating"):
            batch = self._move_to_device(batch)
            
            # Get references
            references = batch.get('raw_transcripts', [])
            all_references.extend(references)
            
            # Generate predictions
            # This is model-specific - adjust based on actual API
            with autocast(enabled=self.config.fp16 or self.config.bf16):
                outputs = self.model.generate(**batch)
                
                # Decode predictions
                if hasattr(self.processor, 'decode'):
                    hypotheses = self.processor.decode(outputs)
                else:
                    hypotheses = outputs  # Adjust based on actual output format
                
                all_hypotheses.extend(hypotheses)
                
                # Compute loss if available
                if hasattr(outputs, 'loss'):
                    total_loss += outputs.loss.item()
                    num_batches += 1
        
        # Compute metrics
        overall_wer = wer(all_references, all_hypotheses)
        overall_cer = cer(all_references, all_hypotheses)
        
        metrics = {
            'wer': overall_wer,
            'cer': overall_cer,
            'num_samples': len(all_references),
        }
        
        if num_batches > 0:
            metrics['loss'] = total_loss / num_batches
        
        logging.info(f"Eval WER: {overall_wer:.2%}, CER: {overall_cer:.2%}")
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard and/or wandb."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        
        if self.config.log_to_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Omnilingual ASR on Bengali dialects")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Override options
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--dialect_filter", type=str, default=None,
                        help="Filter to specific dialect (e.g., 'barishal') or 'all'")
    parser.add_argument("--freeze_encoder", action="store_true", default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint or 'latest'")
    parser.add_argument("--log_to_wandb", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config with overrides
    config = load_config(
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        dialect_filter=args.dialect_filter,
        freeze_encoder=args.freeze_encoder,
        resume_from_checkpoint=args.resume_from_checkpoint,
        log_to_wandb=args.log_to_wandb,
        seed=args.seed,
    )
    
    # Save config
    save_config(config, os.path.join(config.output_dir, "fine_tune_config.yaml"))
    
    # Create trainer and run
    trainer = ASRTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
