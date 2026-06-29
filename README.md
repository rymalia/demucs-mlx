# Demucs MLX

Music source separation on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

A clean MLX port of Meta's [Hybrid Transformer Demucs](https://github.com/facebookresearch/demucs) (HTDemucs) for **inference-only** on Mac M1/M2/M3/M4. Separates any song into 4 stems: **drums**, **bass**, **other**, and **vocals**.

<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple" alt="Apple Silicon MLX">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
</p>

## Performance

Benchmarked on Apple M4 Max with MLX 0.21:

| Audio length | Processing time | Speed |
|:---:|:---:|:---:|
| 10s | 0.4s | **26x** realtime |
| 30s | 1.0s | **30x** realtime |
| 1 min | 1.8s | **33x** realtime |
| 3 min | 5.3s | **34x** realtime |
| 6 min | 10.7s | **34x** realtime |

**11x faster** than PyTorch CPU on the same machine. A 7-minute track separates in about 12 seconds.

Numerical accuracy: max difference vs PyTorch is **< 1 part per million** (0.8 µ). The output is effectively identical.

## Quick start

Demucs MLX requires **Python 3.10–3.13** (numba, a transitive dependency of `librosa`, doesn't support 3.14 yet).

### 1. Install

The easiest way is with [`uv`](https://docs.astral.sh/uv/). It builds an isolated environment, picks a compatible Python automatically, and puts `demucs-mlx` on your PATH:

```bash
uv tool install git+https://github.com/andrade0/demucs-mlx.git

# …or from a local clone:
git clone https://github.com/andrade0/demucs-mlx.git
cd demucs-mlx
uv tool install .
```

> **No uv?** `pipx install git+https://github.com/andrade0/demucs-mlx.git` works the same way. Or, for a plain pip install into your own environment: `pip install git+https://github.com/andrade0/demucs-mlx.git`.

### 2. Separate a song

```bash
demucs-mlx song.mp3
```

The first run downloads the pretrained model (~80 MB, cached in `~/.cache/demucs_mlx/`). Output goes to `separated/htdemucs/<song>/` with 4 WAV files: `drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`.

> If you see a PATH warning during install, run `uv tool update-shell` (or add `~/.local/bin` to your PATH), then restart your terminal.

### Developing on the repo

For local development with live code edits, use the Makefile, which creates a `.venv` via uv, installs the project in editable mode, and links a `demucs-mlx` shim into `~/.local/bin`:

```bash
git clone https://github.com/andrade0/demucs-mlx.git
cd demucs-mlx
make install      # see `make help` for venv / deps / uninstall targets
```

### Usage

```
$ demucs-mlx --help

usage: demucs-mlx [-h] [-n NAME] [-o DIR] [--stems STEM [STEM ...]]
                  [--mp3] [--float32] [--shifts N] [--overlap F]
                  [--no-split] input

Separate a song into stems (drums, bass, other, vocals)
using HTDemucs on Apple Silicon with MLX.

positional arguments:
  input                 input audio file (WAV, MP3, FLAC, OGG, etc.)

model:
  -n, --name NAME       model name (default: htdemucs)

output:
  -o, --output DIR      output directory (default: ./separated/<model>/<song>/)
  --stems STEM [STEM]   stems to save: drums bass other vocals (default: all)
  --mp3                 save as MP3 instead of WAV
  --float32             save as float32 WAV instead of int16

quality:
  --shifts N            random shifts, higher = better but slower (default: 1)
  --overlap F           chunk overlap, 0.0 to 1.0 (default: 0.25)
  --no-split            don't chunk, process whole track at once (more memory)
```

### Examples

```bash
# Extract only vocals
demucs-mlx song.mp3 --stems vocals

# Extract vocals and drums
demucs-mlx song.mp3 --stems vocals drums

# Custom output directory
demucs-mlx song.mp3 -o my_stems/

# Better quality (3 shifts), slower
demucs-mlx song.mp3 --shifts 3

# Use the fine-tuned model
demucs-mlx song.mp3 -n htdemucs_ft

# Save as float32 WAV
demucs-mlx song.mp3 --float32
```

### Use from Python

```python
import mlx.core as mx
import soundfile as sf
from demucs_mlx.pretrained import load_model
from demucs_mlx.apply import apply_model

# Load model (downloads on first use)
model = load_model("htdemucs")

# Load audio
wav, sr = sf.read("song.mp3", dtype="float32")
mix = mx.array(wav.T[None])  # [1, channels, samples]

# Separate
sources = apply_model(model, mix, shifts=1, split=True)
mx.eval(sources)

# sources shape: [1, 4, channels, samples]
# stems: drums, bass, other, vocals
```

## How it works

HTDemucs is a hybrid model that processes audio in two parallel branches:

1. **Frequency branch** — STFT spectrogram through a Conv2d U-Net
2. **Time branch** — Raw waveform through a Conv1d U-Net

Both branches meet at a **Cross-Transformer** bottleneck that lets them exchange information, then decode back to 4 separate source waveforms.

```
                    ┌─────────────────┐
   Waveform ──────►│  STFT (PyTorch)  │
       │            └────────┬────────┘
       │                     │
       ▼                     ▼
  ┌─────────┐         ┌─────────┐
  │  Time   │         │  Freq   │
  │ Encoder │         │ Encoder │
  │ (Conv1d)│         │ (Conv2d)│
  └────┬────┘         └────┬────┘
       │                   │
       └───────┬───────────┘
               ▼
     ┌─────────────────┐
     │ Cross-Transformer│
     │   (5+5 layers)   │
     └────────┬─────────┘
              │
       ┌──────┴──────┐
       ▼              ▼
  ┌─────────┐   ┌─────────┐
  │  Time   │   │  Freq   │
  │ Decoder │   │ Decoder │
  └────┬────┘   └────┬────┘
       │              │
       │         ┌────┴────┐
       │         │  iSTFT  │
       │         └────┬────┘
       └──────┬───────┘
              ▼
      4 separated stems
  (drums, bass, other, vocals)
```

### MLX-specific design choices

- **Layout convention**: Tensors use PyTorch's `[B, C, T]` / `[B, C, H, W]` format externally; transpose at every Conv boundary since MLX expects channels-last
- **STFT/iSTFT**: Runs through PyTorch as a bridge (MLX doesn't support complex tensors). This adds ~5% overhead but keeps the code simple
- **Weight loading**: Downloads the original PyTorch checkpoint, converts Conv weights (transpose axes) and splits MultiheadAttention projections, then loads into the MLX model
- **No training code**: Inference-only, all batch norm / dropout / weight init stripped out

## Choosing a model

Models download automatically on first use (~80 MB each, cached in `~/.cache/demucs_mlx/`). Use the `-n` flag to switch models.

### Available models

| Model | Sources | Description | Status |
|-------|---------|-------------|--------|
| `htdemucs` | 4 (drums, bass, other, vocals) | Default HTDemucs — fast and reliable | Supported |
| `htdemucs_ft` | 4 (drums, bass, other, vocals) | Fine-tuned on more data — better vocal separation | Supported |
| `htdemucs_6s` | 6 (drums, bass, other, vocals, guitar, piano) | 6-source variant — separates guitar and piano too | Untested |

### Which model should I use?

- **General use?** Start with `htdemucs` (the default). It's the standard model and works well on most music.
- **Best vocal isolation?** Use `htdemucs_ft` — it's fine-tuned on a larger dataset and produces cleaner vocal separation, especially on complex mixes.
- **Need guitar or piano stems?** Try `htdemucs_6s` — it outputs 6 separate stems instead of 4. Note: this variant is untested in the MLX port, so results may vary.

```bash
# Default model (4 stems)
demucs-mlx song.mp3

# Fine-tuned model (better vocals)
demucs-mlx song.mp3 -n htdemucs_ft

# 6-source model (adds guitar + piano)
demucs-mlx song.mp3 -n htdemucs_6s
```

## Requirements

- **macOS** with Apple Silicon (M1, M2, M3, or M4)
- **Python 3.10–3.13** (3.14 not yet supported — `librosa`→`numba` has no 3.14 wheels)
- **RAM**: 8 GB minimum, 16 GB+ recommended for long tracks without chunking
- [`uv`](https://docs.astral.sh/uv/) or `pipx` recommended for an isolated install (avoids dependency conflicts)

### Dependencies

Installed automatically by `make install`:

| Package | Why |
|---------|-----|
| `mlx` >= 0.17 | Apple's ML framework (GPU acceleration) |
| `torch` | STFT/iSTFT + weight loading (CPU only) |
| `numpy` | Array operations |
| `soundfile` | Audio file I/O (WAV, MP3, FLAC, OGG) |
| `pyyaml` | Model config parsing |
| `tqdm` | Progress bars |

Optional: `librosa` (only needed if your input audio isn't 44.1 kHz — it handles resampling).

> **Note on PyTorch**: PyTorch is only used for STFT/iSTFT and loading pretrained weights. It runs on CPU — all the heavy computation (convolutions, transformer, U-Net) runs on Apple GPU through MLX. A future version may remove the PyTorch dependency entirely.

## Project structure

```
demucs-mlx/
├── Makefile                 # editable dev install (make install / help)
├── pyproject.toml           # package metadata + `demucs-mlx` entry point
└── demucs_mlx/
    ├── cli.py               # CLI entry point (demucs-mlx / python -m demucs_mlx)
    ├── __main__.py          # enables `python -m demucs_mlx`
    ├── htdemucs.py          # Main model (hybrid U-Net + transformer)
    ├── hdemucs.py           # Encoder/decoder layers
    ├── transformer.py       # Cross-attention transformer
    ├── demucs.py            # DConv residual branches
    ├── apply.py             # Inference pipeline (chunking, shifts)
    ├── pretrained.py        # Model download & weight loading
    ├── weight_convert.py    # PyTorch → MLX weight conversion
    ├── spec.py              # STFT/iSTFT bridge
    ├── utils.py             # Tensor utilities
    └── remote/              # Model download configs (shipped in the wheel)
        ├── files.txt
        └── htdemucs.yaml
```

## License

MIT License — same as the [original Demucs](https://github.com/facebookresearch/demucs).

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research — the original PyTorch model and pretrained weights
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework that makes this fast on Apple Silicon
- HTDemucs paper: [Rouard, Massa, Défossez (2023)](https://arxiv.org/abs/2211.08553)
