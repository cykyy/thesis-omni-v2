# Creating manifest files

`src/create_manifest.py` turns the RegSpeech12 Excel transcripts and audio into manifest
files for fairseq2's manifest dataset loader.

```bash
python src/create_manifest.py --output_dir data/regspeech12_manifest --format tsv
```

**Expected output:**

```
data/regspeech12_manifest/
├── train.tsv    (~80 hours)
├── valid.tsv    (~10 hours)
├── test.tsv     (~10 hours)
└── README.txt
```
