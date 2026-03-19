import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from miditok import REMI, TokenizerConfig
from transformers import AutoTokenizer, AutoModel

from midilm.model import MidiLM, DualPathMidiLM
from config.midilm_config import MidiLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the MIDI to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--midilm_ckpt", type=str, required=True, help="Path to the MidiLM checkpoint")
    parser.add_argument("--midi_max_len", type=int, default=2048)
    parser.add_argument("--prompt_max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        mixed_precision="bf16"
    )

    config = TokenizerConfig(
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        one_token_stream_for_programs=True,
        special_tokens=['PAD', 'BOS', 'EOS']
    )
    midi_tokenizer = REMI(config)
    prompt_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    midilm_config = MidiLMConfig(vocab_size=len(midi_tokenizer.vocab))
    midilm = MidiLM(midilm_config)
    prompt_encoder = AutoModel.from_pretrained("openai-community/gpt2")
    model = DualPathMidiLM(midilm, prompt_encoder)
    model.load_state_dict(torch.load(args.midilm_ckpt))

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Total parameters: {total_params} (approx {total_params / 1e9:.2f}B)")

    model = accelerator.prepare(model)
    model.eval()

    # Encode text prompt
    prompt_ids = prompt_tokenizer.encode(args.prompt, add_special_tokens=False)
    if len(prompt_ids) > args.prompt_max_len:
        prompt_ids = prompt_ids[:args.prompt_max_len]
    else:
        prompt_ids += [prompt_tokenizer.eos_token_id] * (args.prompt_max_len - len(prompt_ids))
    prompt_ids = torch.LongTensor([prompt_ids]).to(accelerator.device)

    # BOS token as initial input
    input_ids = torch.LongTensor([[1]]).to(accelerator.device)
    max_generated_tokens = args.midi_max_len - 1

    accelerator.print(f"Prompt: {args.prompt}")
    accelerator.print(f"Generating MIDI with temperature={args.temperature}, max_len={max_generated_tokens}...")

    generated_sequence_ids, _, _ = model.generate(
        prompt_input_ids=prompt_ids,
        init_input_ids=input_ids,
        max_new_tokens=max_generated_tokens,
        temperature=args.temperature,
    )

    generated_midi = midi_tokenizer(generated_sequence_ids.squeeze(0).cpu().numpy())
    midi_path = Path(args.output_dir, "generated.mid")
    generated_midi.dump_midi(midi_path)
    accelerator.print(f"MIDI saved to: {midi_path}")


if __name__ == "__main__":
    main()
