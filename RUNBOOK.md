# VLM KIE — Runbook

Quick reference for running, testing, and extending this project.

---

## Environment

- Python 3.11, managed by `uv` (never use `pip` directly)
- PyTorch 2.4.1+cu121 (compatible with driver 581.80 / CUDA 12.4)
- WSL2 quirk: `import torch` segfaults via `.venv/bin/python` directly — always use `uv run`

```bash
# Verify environment
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.1+cu121 True
```

---

## Run Commands

### Single image extraction
```bash
uv run run.py --model qwen3.5-0.8b --image path/to/invoice.png
uv run run.py --model glm-ocr --image path/to/invoice.png
```

### Full benchmark (all models, 100 CORD-v2 samples)
```bash
uv run run.py --model all --dataset cord-v2 --n 100
```

### Limited sanity check (10 samples)
```bash
uv run run.py --model all --dataset cord-v2 --n 10
```

### View comparison table
```bash
uv run scripts/compare_results.py --run outputs/run_YYYYMMDD_HHMMSS/
```

---

## Tests

```bash
uv run pytest                    # all tests
uv run pytest -v                 # verbose
uv run pytest tests/test_metrics.py   # specific file
```

Currently: **35/35 tests passing** (`test_json_repair.py`, `test_metrics.py`)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_samples.py` | Download CORD-v2 test split from HuggingFace |
| `scripts/run_benchmark.py` | Full CLI benchmark runner |
| `scripts/compare_results.py` | Render `outputs/` run → markdown table |

```bash
uv run scripts/download_samples.py --n 100
uv run scripts/run_benchmark.py --model qwen3.5-2b --dataset cord-v2 --n 50
uv run scripts/compare_results.py --run outputs/run_20260319_143000/
```

---

## Dependencies

```bash
# Add a package
# 1. Add to [project.dependencies] in pyproject.toml
# 2. Then:
uv sync

# Install optional paddle deps (PP-ChatOCRv4)
uv sync --extra paddle
```

---

## Next Steps (as of 2026-03-19)

The scaffold is complete and all unit tests pass. Outstanding work in order:

### Step 1 — Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
# Verify:
ollama --version
```

### Step 2 — Pull Qwen3.5 models
```bash
ollama pull qwen3.5:0.8b   # 1.0 GB disk, <2 GB VRAM — start here
ollama pull qwen3.5:2b     # 2.7 GB disk, ~3 GB VRAM
ollama pull qwen3.5:4b     # 3.4 GB disk, ~4 GB VRAM
```

### Step 3 — Smoke test Qwen3.5
```bash
# Need a sample image first — use any invoice PNG, or download CORD-v2 samples:
uv run scripts/download_samples.py --n 5

uv run run.py --model qwen3.5-0.8b --image src/vlm_kie/data/samples/cord-v2/test_0.png
```
Expected: JSON output with invoice fields printed to stdout, `outputs/` directory created.

### Step 4 — Download full CORD-v2 test split
```bash
uv run scripts/download_samples.py --n 100
# Saves to: src/vlm_kie/data/samples/cord-v2/
```

### Step 5 — Multi-model sanity check
```bash
uv run run.py --model all --dataset cord-v2 --n 10
uv run scripts/compare_results.py --run outputs/<latest-run>/
```

### Step 6 — GLM-OCR end-to-end
```bash
uv run run.py --model glm-ocr --image src/vlm_kie/data/samples/cord-v2/test_0.png
# Downloads zai-org/GLM-OCR (~1.7GB) from HuggingFace on first run
```

### Step 7 — PaddleOCR-VL-1.5 end-to-end
```bash
uv run run.py --model paddleocr-vl-1.5 --image src/vlm_kie/data/samples/cord-v2/test_0.png
# Downloads PaddlePaddle/PaddleOCR-VL-1.5 (~4GB, 4-bit quantized) on first run
# No paddlepaddle package needed — uses transformers backend
```

### Step 8 — PP-ChatOCRv4 (optional, requires paddle)
```bash
uv sync --extra paddle   # installs paddlepaddle, paddleocr, paddlex (~large)
uv run run.py --model pp-chatocrv4 --image src/vlm_kie/data/samples/cord-v2/test_0.png
```

### Step 9 — Full benchmark
```bash
uv run run.py --model all --dataset cord-v2 --n 100
# Results in: outputs/run_YYYYMMDD_HHMMSS/{model}/results.jsonl + metrics.json
# Summary: outputs/run_YYYYMMDD_HHMMSS/comparison.md
```

---

## Outputs Layout

```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── qwen3.5-0.8b/
    │   ├── results.jsonl    # one ExtractionResult per line
    │   └── metrics.json     # aggregated exact_match / token_f1 / partial_match
    ├── glm-ocr/
    │   └── ...
    └── comparison.md        # side-by-side model comparison table
```

---

## Model VRAM Reference

| Model | Backend | VRAM | Notes |
|-------|---------|------|-------|
| qwen3.5:0.8b | Ollama | <2 GB | Fastest, lowest quality |
| qwen3.5:2b | Ollama | ~3 GB | Best Qwen3.5 option |
| qwen3.5:4b | Ollama | ~4 GB | Max size for 1050 Ti |
| GLM-OCR | transformers fp16 | ~1.5 GB | Document-specific |
| PaddleOCR-VL-1.5 | transformers 4-bit | ~4 GB | Large model, quantized |
| PP-ChatOCRv4 | PaddleX + Ollama | ~3 GB | Hybrid OCR pipeline |

---

## Server Replication

```bash
# On server (after git clone):
uv sync
uv run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Ollama + pull models
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3.5:2b

# Run benchmark
uv run run.py --model all --dataset cord-v2 --n 100
```
