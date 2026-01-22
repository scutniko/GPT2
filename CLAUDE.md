# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Rules

**IMPORTANT**: Always output your thinking process and results in **Chinese** (中文). This applies to all explanations, analysis, and responses when working with this codebase.

## Project Overview

This is a modular GPT-2 training framework for ablation studies. It supports various attention mechanisms, position encodings, MLP variants, activation functions, and normalization layers. All components are pluggable through experiment configurations.

**Key characteristics:**
- Unified training entry point (`train.py`) for all experiments
- Fully rename-friendly - project folder can be renamed without code changes
- Dynamic path management - all paths auto-adapt, cross-platform compatible
- Automatic multi-GPU DDP training with GPU detection
- Smart checkpoint recovery

## Common Commands

### Training
```bash
# Single GPU
python train.py --experiment baseline

# Multi-GPU (recommended)
torchrun --standalone --nproc_per_node=8 train.py --experiment baseline

# Automated training with auto-resume (recommended)
bash train_nightly.sh
EXPERIMENT=rope bash train_nightly.sh
EXPERIMENT=baseline bash train_nightly.sh

# Resume from checkpoint
python train.py --experiment baseline --resume log_train/baseline/log/model_05000.pt

# Inference mode
python train.py --experiment baseline --inference log_train/baseline/log/model_05000.pt
```

### Monitoring
```bash
# View logs
tail -f log_train/baseline/log/log.txt

# Plot training curves
python plot_training_log.py log_train/baseline/log/log.txt curves.png 1000
```

## Architecture

### Module Injection Pattern

The framework uses a configuration-based module injection system. Each experiment in `experiments/*.py` specifies:

1. `MODEL_CONFIG` - GPTConfig with model hyperparameters
2. `ATTENTION_CLASS` - Attention mechanism class (BaseAttention, MQAAttention, GQAAttention, MLAAttention)
3. `POSITION_ENCODING_CLASS` - Position encoding class (LearnedPositionEncoding, ALiBi, RoPE, SinusoidalPositionalEncoding)
4. `MLP_CLASS` (optional) - MLP variant
5. `NORM_CLASS` (optional) - Normalization layer (nn.LayerNorm, RMSNorm)
6. `TRAINING_CONFIG` - Training hyperparameters

### Component Flow

```
train.py
  ├─ Loads experiment config from experiments/*.py
  ├─ Creates GPT(model_config, attention_class, position_encoding_class, mlp_class, norm_class)
  │   └─ GPT dynamically assembles:
  │       ├─ Token embeddings (wte)
  │       ├─ Position encoding (wpe/alibi/rope/sine - based on class)
  │       ├─ Transformer blocks (n_layer × Block)
  │       │   └─ Each Block contains: ln_1 → attn → ln_2 → mlp
  │       ├─ Final normalization (ln_f)
  │       └─ LM head (weight-tied with wte)
  └─ Training loop with periodic HellaSwag evaluation
```

### Key Design Patterns

**Position Encoding Injection**: Position encodings are injected either at the model level (learned/sine) or at the attention level (ALiBi, RoPE). The GPT constructor detects which type is used and passes `rope` or `alibi` objects to each Block.

**Weight Tying**: `self.transformer.wte.weight = self.lm_head.weight` - token embeddings and output layer share weights.

**Module Defaults**: When creating new experiments, you only need to specify the components you're changing. Unspecified components use sensible defaults (BaseAttention, LearnedPositionEncoding, standard MLP, LayerNorm).

## Supported Experiments

| Category | Experiments | Description |
|----------|-------------|-------------|
| Position Encoding | `baseline`, `alibi`, `rope`, `sine` | Various position encoding schemes |
| Attention | `mqa`, `gqa`, `mla` | Multi-Query, Grouped Query, Multi-head Latent Attention |
| Normalization | `rmsnorm` | RMSNorm instead of LayerNorm |
| Activation | `geglu`, `swiglu`, `silu`, `relu` | Different MLP activation functions |

## Creating New Experiments

1. Copy an existing config: `cp experiments/baseline.py experiments/my_experiment.py`
2. Edit the configuration classes
3. Run: `python train.py --experiment my_experiment`

## Important Files

- `train.py` - Main training entry point, handles DDP, checkpointing, evaluation
- `models/gpt.py` - GPT model with pluggable components
- `modules/block.py` - Transformer block that accepts injected classes
- `core/config.py` - GPTConfig dataclass
- `experiments/*.py` - Experiment configurations

## Training Details

- Default vocab_size is 50304 (not 50257) for better GPU alignment
- 19,073 steps ≈ 1 epoch on ~10B tokens with 0.5M token batch size
- Validation loss and HellaSwag accuracy evaluated every 250 steps
- Checkpoints saved at evaluation intervals
- Text generation samples produced every 250 steps for monitoring
