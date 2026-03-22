# MidiLM

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://doi.org/10.1609/aaai.v40i28.39483)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Large-Multimodal-Model-Lab/MIDILM)

[Demo](https://large-multimodal-model-lab.github.io/MIDILM/)


Official inference code for the paper **"MIDILM: A Dual-Path Model for Controllable Text-to-MIDI Generation"** (AAAI 2026).

MidiLM is a 1.67B-parameter language model for text-to-MIDI generation. It employs a dual-path architecture with a Mixture of Experts (MoE) decoder, conditioned on text prompts via a pretrained GPT-2 encoder.

## Architecture

- **MIDI Decoder**: 12-layer Transformer with Grouped Query Attention (GQA) and RoPE, using Mixture of Experts (8 experts, top-2 routing) as the feed-forward network
- **Text Encoder**: Pretrained GPT-2 with a linear projection layer
- **Dual-Path Design**: Text and MIDI hidden states are concatenated for shared self-attention, then split for separate feed-forward processing (MLP for text, MoE for MIDI)
- **Tokenizer**: REMI representation via MidiTok

## Project Structure

```
├── inference.py              # Inference entry point
├── config/
│   └── midilm_config.py      # Model configuration
├── midilm/
│   └── model.py              # Model definition (MidiLM, DualPathMidiLM)
└── demo/
    └── sample_*.wav            # 5 demo samples
```

## Requirements

- Python >= 3.10
- CUDA >= 12.0 (recommended)

```bash
pip install -r requirements.txt
```

> **Important**: `miditok` must be version `3.0.5.post1`. Other versions produce different vocabulary sizes, causing checkpoint incompatibility.

## Inference

```bash
python inference.py \
  --prompt "A melodic pop song with electronic elements, featuring acoustic guitar, piano, synth brass, clean electric guitar, and harmonica, all contributing to a festive Christmas atmosphere. Set in the key of Bb major with a moderate tempo of 105 bpm, the piece maintains a 4/4 time signature throughout its duration. The recurring chord progression of Bb, Gm, and Eb forms the harmonic foundation, creating a meditative and uplifting mood." \
  --midilm_ckpt /path/to/unwrap_model.pt \
  --output_dir ./output \
  --temperature 1.0 \
  --midi_max_len 2048 \
  --seed 42
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompt` | (required) | Text description of the MIDI to generate |
| `--midilm_ckpt` | (required) | Path to the model checkpoint |
| `--output_dir` | `./output` | Directory to save generated MIDI |
| `--temperature` | `1.0` | Sampling temperature |
| `--midi_max_len` | `2048` | Maximum MIDI sequence length |
| `--prompt_max_len` | `128` | Maximum text prompt token length |
| `--seed` | `42` | Random seed |

The generated MIDI file will be saved as `generated.mid` in the output directory.

## Demo Samples

| # | Prompt | Audio |
|---|--------|-------|
| 1 | A melodic pop song with electronic elements, featuring acoustic guitar, piano, synth brass, clean electric guitar, and harmonica, all contributing to a festive Christmas atmosphere. | [▶ Play](demo/sample_1.wav) |
| 2 | A melodic electronic song with a spacey and dreamy atmosphere, featuring synth strings, drums, electric bass, glockenspiel, and a brass section. | [▶ Play](demo/sample_2.wav) |
| 3 | A cheerful pop Christmas song in D minor, featuring electric and acoustic guitars, trumpet, trombone, and pan flute. | [▶ Play](demo/sample_3.wav) |
| 4 | A melodic and energetic rock song with electronic elements, featuring distorted guitars, electric bass, synth strings, alto saxophone, and synth voice. | [▶ Play](demo/sample_4.wav) |
| 5 | A melodic classical and electronic piece featuring piano, violin, and cello, set in the key of G major with a fast tempo of 144 bpm. | [▶ Play](demo/sample_5.wav) |

## Model Checkpoint

The pretrained checkpoint can be downloaded from: [TBD]

## Citation

```bibtex
@article{li2026midilm,
  title={MIDILM: A Dual-Path Model for Controllable Text-to-MIDI Generation},
  author={Li, Shuyu and Choi, Dooho and Sung, Yunsick},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={28},
  pages={23160--23168},
  year={2026},
  doi={10.1609/aaai.v40i28.39483}
}
```

