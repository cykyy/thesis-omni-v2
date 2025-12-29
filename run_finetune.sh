#!/bin/bash
# Fine-tuning Omnilingual ASR on RegSpeech12
# All files stay in /root/thesis - nothing copied to omnilingual-asr repo

set -e

echo "=============================================="
echo "Fine-tuning Omnilingual ASR on RegSpeech12"
echo "=============================================="

# ============================================
# CONFIGURATION - Edit these paths if needed
# ============================================
THESIS_DIR="/root/thesis"
OMNI_DIR="/root/omnilingual-asr"
OUTPUT_DIR="${THESIS_DIR}/checkpoints/llm_finetune_v1"
PARQUET_DIR="${THESIS_DIR}/data/regspeech12_parquet"

# ============================================
# STEP 1: Set environment variable for asset cards
# ============================================
echo ""
echo "Step 1: Setting FAIRSEQ2_USER_ASSET_DIR..."
export FAIRSEQ2_USER_ASSET_DIR="${THESIS_DIR}/cards"
echo "  FAIRSEQ2_USER_ASSET_DIR=${FAIRSEQ2_USER_ASSET_DIR}"

# Add to bashrc for persistence (optional)
# echo 'export FAIRSEQ2_USER_ASSET_DIR="/root/thesis/cards"' >> ~/.bashrc

# ============================================
# STEP 2: Convert RegSpeech12 to Parquet (if not done)
# ============================================
if [ ! -d "${PARQUET_DIR}/corpus=regspeech12" ]; then
    echo ""
    echo "Step 2: Converting RegSpeech12 to Parquet..."
    cd ${THESIS_DIR}
    python convert_to_parquet_v2.py --output_dir ${PARQUET_DIR} --batch_size 200
else
    echo ""
    echo "Step 2: Parquet data already exists, skipping conversion..."
fi

# ============================================
# STEP 3: Create output directory
# ============================================
echo ""
echo "Step 3: Creating output directory..."
mkdir -p ${OUTPUT_DIR}

# ============================================
# STEP 4: Run fine-tuning
# ============================================
echo ""
echo "Step 4: Starting fine-tuning..."
echo "  Config: ${THESIS_DIR}/configs/finetune_config.yaml"
echo "  Output: ${OUTPUT_DIR}"

cd ${OMNI_DIR}
python -m workflows.recipes.wav2vec2.asr ${OUTPUT_DIR} \
    --config-file ${THESIS_DIR}/configs/finetune_config.yaml

echo ""
echo "=============================================="
echo "Fine-tuning complete!"
echo "Checkpoints: ${OUTPUT_DIR}"
echo "=============================================="
