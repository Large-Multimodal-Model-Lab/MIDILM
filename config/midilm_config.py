from dataclasses import dataclass

@dataclass
class MidiLMConfig:
    vocab_size: int = 515
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_heads: int = 16
    num_key_value_heads: int = 4
    attn_dropout_prob = 0.0
    moe_dropout_prob = 0.0
    max_position_embeddings = 2048
    rope_theta = 10000.0
    num_local_experts = 8
    num_experts_per_tok = 2
    load_balancing_alpha = 0.01
    pad_token_id = 0

midilm_base_config = MidiLMConfig()
