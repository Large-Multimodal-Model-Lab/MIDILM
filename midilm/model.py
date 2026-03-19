import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    b, nk, sl, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nk, n_rep, sl, hd)
    return hidden_states.reshape(b, nk * n_rep, sl, hd)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        inp_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(inp_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        dt = x.device.type
        dt = dt if isinstance(dt, str) and dt != "mps" else "cpu"
        with torch.autocast(device_type=dt, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_key_value_heads, attn_dropout_porb, max_position_embeddings, rope_theta):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.key_value_size = self.head_dim * num_key_value_heads
        self.attn_dropout_prob = attn_dropout_porb
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_value_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.key_value_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)

    def forward(self, hidden_states, output_attentions=False):
        bsz, q_len, _ = hidden_states.size()
        position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = None
        if output_attentions:
            attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            if self.training and self.attn_dropout_prob > 0:
                attn_weights = F.dropout(attn_weights, p=self.attn_dropout_prob)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True,
                    dropout_p=self.attn_dropout_prob if self.training else 0.0,
                )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights
        return attn_output

class MLP(nn.Module):
    def __init__(self, hidden_size, inter):
        super().__init__()
        self.hidden_size = hidden_size
        self.inter_dim = inter
        self.gate_proj = nn.Linear(self.hidden_size, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.hidden_size, bias=False)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x

class MoeLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok,
                 moe_dropout_prob, load_balancing_alpha):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_dropout_prob = moe_dropout_prob
        self.experts = nn.ModuleList([
            MLP(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_local_experts)
        ])
        self.gate = nn.Linear(self.hidden_size, self.num_local_experts, bias=False)
        self.dropout = nn.Dropout(self.moe_dropout_prob)
        self.load_balancing_alpha = load_balancing_alpha

    def forward(self, h):
        b, l, d = h.shape
        gate_logits = self.gate(h)
        results = torch.zeros_like(h)
        topk_storage = []
        for j in range(b):
            weights, selected_experts = torch.topk(gate_logits[j], self.num_experts_per_tok, dim=-1)
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(h.dtype)
            topk_storage.append((weights, selected_experts))
            for k, expert in enumerate(self.experts):
                batch_idx, nth_expert = torch.where(selected_experts == k)
                expert_output = expert(h[j][batch_idx])
                results[j][batch_idx] += weights[batch_idx, nth_expert, None] * expert_output
        results = self.dropout(results)
        route_probs_all = torch.zeros(b, l, self.num_local_experts, device=h.device, dtype=h.dtype)
        for j in range(b):
            w_j, idx_j = topk_storage[j]
            flat_inds = idx_j.reshape(-1)
            flat_vals = w_j.reshape(-1)
            row_idx = torch.arange(l, device=h.device).unsqueeze(-1)
            row_idx = row_idx.expand(l, self.num_experts_per_tok).reshape(-1)
            route_probs_all[j].index_put_((row_idx, flat_inds), flat_vals, accumulate=True)
        p_i = route_probs_all.mean(dim=(0, 1))
        lb_loss = self.load_balancing_alpha * (p_i * p_i).sum()
        expert_routing_info = {
            'route_probs': route_probs_all,
            'topk_experts': [item[1] for item in topk_storage],
            'topk_weights': [item[0] for item in topk_storage]
        }
        return results, lb_loss, expert_routing_info

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.moe_norm = RMSNorm(config.hidden_size)
        self.self_attn = SelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_key_value_heads,
            attn_dropout_porb=config.attn_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.moe = MoeLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_local_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_dropout_prob=config.moe_dropout_prob,
            load_balancing_alpha=config.load_balancing_alpha
        )

    def forward(self, hidden_states, output_attentions=False):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        if output_attentions:
            attn_out, attn_weights = self.self_attn(hidden_states, output_attentions=True)
        else:
            attn_out = self.self_attn(hidden_states)

        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.moe_norm(hidden_states)
        moe_out, lb_loss, expert_routing_info = self.moe(hidden_states)
        hidden_states = residual + moe_out

        if output_attentions:
            return hidden_states, lb_loss, expert_routing_info, attn_weights
        return hidden_states, lb_loss, expert_routing_info

class MidiLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([Decoder(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.vocab_size, config.hidden_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids, output_attentions=False):
        hidden_states = self.token_embedding(input_ids)
        lb_losses = []
        all_expert_routing_info = []
        all_attention_weights = [] if output_attentions else None

        for i, layer in enumerate(self.layers):
            if output_attentions:
                hidden_states, lb_loss, expert_routing_info, attn_weights = layer(hidden_states, output_attentions=True)
                all_attention_weights.append(attn_weights)
            else:
                hidden_states, lb_loss, expert_routing_info = layer(hidden_states)

            lb_losses.append(lb_loss)
            all_expert_routing_info.append(expert_routing_info)

        hidden_states = self.norm(hidden_states)
        token_logits = self.lm_head(hidden_states)
        total_lb_loss = torch.stack(lb_losses).sum()

        if output_attentions:
            return token_logits, total_lb_loss, all_expert_routing_info, all_attention_weights
        return token_logits, total_lb_loss, all_expert_routing_info

class DualPathMidiLM(nn.Module):
    def __init__(self, midilm, prompt_encoder):
        super().__init__()
        self.midilm = midilm
        self.prompt_encoder = prompt_encoder
        self.n_layers = len(midilm.layers)
        self.hidden_size = midilm.hidden_size
        self.prompt_hidden_size = self.prompt_encoder.config.hidden_size
        self.proj = nn.Linear(self.prompt_hidden_size, self.hidden_size, bias=False)

        self.prompt_mlps = nn.ModuleList([
            MLP(self.hidden_size, int(self.hidden_size*8))
            for _ in range(self.n_layers)
        ])

        self.prompt_norms = nn.ModuleList([
            RMSNorm(self.hidden_size)
            for _ in range(self.n_layers)
        ])

    def forward(self, input_ids, prompt_input_ids, output_attentions=False):
        prompt_outputs = self.prompt_encoder(prompt_input_ids)
        prompt_hidden_states = prompt_outputs.last_hidden_state
        prompt_hidden_states = self.proj(prompt_hidden_states)
        hidden_states = self.midilm.token_embedding(input_ids)
        lb_losses = []
        all_expert_routing_info = []
        all_attention_weights = [] if output_attentions else None

        if output_attentions:
            self.prompt_len = prompt_hidden_states.size(1)

        for layer_idx, layer in enumerate(self.midilm.layers):
            cat_input = torch.cat([prompt_hidden_states, hidden_states], dim=1)
            residual = cat_input

            normalized_cat_input = layer.attn_norm(cat_input)

            if output_attentions:
                attn_out, attn_weights = layer.self_attn(normalized_cat_input, output_attentions=True)
                all_attention_weights.append(attn_weights)
            else:
                attn_out = layer.self_attn(normalized_cat_input)

            cat_hidden_states = residual + attn_out

            prompt_len = prompt_hidden_states.size(1)
            curr_prompt_hidden = cat_hidden_states[:, :prompt_len, :]
            curr_midi_hidden = cat_hidden_states[:, prompt_len:, :]

            prompt_residual = curr_prompt_hidden
            normalized_prompt = self.prompt_norms[layer_idx](curr_prompt_hidden)
            prompt_mlp_out = self.prompt_mlps[layer_idx](normalized_prompt)
            curr_prompt_hidden = prompt_residual + prompt_mlp_out

            midi_residual = curr_midi_hidden
            normalized_midi = layer.moe_norm(curr_midi_hidden)
            moe_out, lb_loss, expert_routing_info = layer.moe(normalized_midi)
            curr_midi_hidden = midi_residual + moe_out

            prompt_hidden_states = curr_prompt_hidden
            hidden_states = curr_midi_hidden

            lb_losses.append(lb_loss)
            all_expert_routing_info.append(expert_routing_info)

        hidden_states = self.midilm.norm(hidden_states)
        logits = self.midilm.lm_head(hidden_states)
        total_lb_loss = torch.stack(lb_losses).sum()

        if output_attentions:
            return logits, total_lb_loss, all_expert_routing_info, all_attention_weights
        return logits, total_lb_loss, all_expert_routing_info

    @torch.no_grad()
    def generate(self,
                 prompt_input_ids,
                 init_input_ids,
                 max_new_tokens,
                 temperature=1.0,
                 output_attentions=False):
        self.eval()
        generated = init_input_ids.clone()
        all_token_expert_info = []
        all_token_attention_maps = [] if output_attentions else None
        generated_tokens = []

        for i in range(max_new_tokens):
            if output_attentions:
                logits, _, expert_routing_info, attention_weights = self.forward(
                    generated,
                    prompt_input_ids,
                    output_attentions=True
                )
                all_token_attention_maps.append(attention_weights)
            else:
                logits, _, expert_routing_info = self.forward(
                    generated,
                    prompt_input_ids
                )

            all_token_expert_info.append(expert_routing_info)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            generated = torch.cat([generated, next_token], dim=1)

        if output_attentions:
            prompt_len = getattr(self, 'prompt_len', 0)
            return generated, all_token_expert_info, generated_tokens, all_token_attention_maps, prompt_len
        return generated, all_token_expert_info, generated_tokens
