import torch
from src.arc.model import GPTModel
from transformers import GPT2LMHeadModel
import tiktoken


def load_hf_gpt2_weights(custom_model: GPTModel, model_name="gpt2"):
    """
    Loads Hugging Face GPT-2 weights into the custom GPTModel architecture.
    """
    print(f"Loading '{model_name}' weights from Hugging Face...")

    # 1. Load the official Hugging Face model
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()
    custom_sd = custom_model.state_dict()

    # 2. Extract configuration constants
    n_layers = custom_model.trf_blocks.__len__()
    emb_dim = custom_model.tok_emb.embedding_dim

    with torch.no_grad():
        # --- Embeddings & Final LayerNorm ---
        # Token and Positional Embeddings
        custom_model.tok_emb.weight.copy_(hf_sd["transformer.wte.weight"])
        custom_model.pos_emb.weight.copy_(hf_sd["transformer.wpe.weight"])

        # Tie the LM head weights to the token embeddings
        custom_model.out_head.weight.copy_(hf_sd["transformer.wte.weight"])

        # Final LayerNorm (Custom model uses 'scale' and 'shift' instead of 'weight' and 'bias')
        custom_model.final_norm.scale.copy_(hf_sd["transformer.ln_f.weight"])
        custom_model.final_norm.shift.copy_(hf_sd["transformer.ln_f.bias"])

        # --- Transformer Blocks ---
        for b in range(n_layers):
            # 1. Block LayerNorms
            custom_model.trf_blocks[b].norm1.scale.copy_(
                hf_sd[f"transformer.h.{b}.ln_1.weight"]
            )
            custom_model.trf_blocks[b].norm1.shift.copy_(
                hf_sd[f"transformer.h.{b}.ln_1.bias"]
            )

            custom_model.trf_blocks[b].norm2.scale.copy_(
                hf_sd[f"transformer.h.{b}.ln_2.weight"]
            )
            custom_model.trf_blocks[b].norm2.shift.copy_(
                hf_sd[f"transformer.h.{b}.ln_2.bias"]
            )

            # 2. Attention (Fused QKV -> Separate Q, K, V)
            # HF shape: (emb_dim, 3 * emb_dim). We transpose to (3 * emb_dim, emb_dim) and split.
            c_attn_weight = hf_sd[f"transformer.h.{b}.attn.c_attn.weight"].t()
            q_w, k_w, v_w = c_attn_weight.split(emb_dim, dim=0)

            custom_model.trf_blocks[b].att.W_query.weight.copy_(q_w)
            custom_model.trf_blocks[b].att.W_key.weight.copy_(k_w)
            custom_model.trf_blocks[b].att.W_value.weight.copy_(v_w)

            # Split biases for Q, K, V
            c_attn_bias = hf_sd[f"transformer.h.{b}.attn.c_attn.bias"]
            q_b, k_b, v_b = c_attn_bias.split(emb_dim, dim=0)

            custom_model.trf_blocks[b].att.W_query.bias.copy_(q_b)
            custom_model.trf_blocks[b].att.W_key.bias.copy_(k_b)
            custom_model.trf_blocks[b].att.W_value.bias.copy_(v_b)

            # Attention output projection
            c_proj_weight = hf_sd[f"transformer.h.{b}.attn.c_proj.weight"].t()
            custom_model.trf_blocks[b].att.out_proj.weight.copy_(c_proj_weight)
            custom_model.trf_blocks[b].att.out_proj.bias.copy_(
                hf_sd[f"transformer.h.{b}.attn.c_proj.bias"]
            )

            # 3. Feed Forward Network (MLP)
            # HF uses c_fc for the first layer and c_proj for the second
            mlp_c_fc_weight = hf_sd[f"transformer.h.{b}.mlp.c_fc.weight"].t()
            custom_model.trf_blocks[b].ff.layers[0].weight.copy_(mlp_c_fc_weight)
            custom_model.trf_blocks[b].ff.layers[0].bias.copy_(
                hf_sd[f"transformer.h.{b}.mlp.c_fc.bias"]
            )

            mlp_c_proj_weight = hf_sd[f"transformer.h.{b}.mlp.c_proj.weight"].t()
            custom_model.trf_blocks[b].ff.layers[2].weight.copy_(mlp_c_proj_weight)
            custom_model.trf_blocks[b].ff.layers[2].bias.copy_(
                hf_sd[f"transformer.h.{b}.mlp.c_proj.bias"]
            )

    print("Weights successfully loaded!")
    return custom_model


def generate_text(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    text: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cpu",
):
    encoded = tokenizer.encode(text)
    tokenized = tokenizer.decode_tokens_bytes(encoded)
    print(f"tokenized input: {tokenized} | number of tokens: {len(encoded)}")
    print(f"encoded input: {encoded}")

    in_idx = torch.tensor(encoded).unsqueeze(0).to(device)  # Shape: (1, seq_len)

    model.eval()
    for _ in range(max_new_tokens):
        in_idx_cond = in_idx[
            :, -context_length:
        ]  # Crop context to the model's maximum context length

        with torch.inference_mode():
            logits = model(in_idx_cond)

        # 1. Focus on the last step and apply temperature
        logits = logits[:, -1, :] / temperature

        # 2. Optional: Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # 3. Convert to probabilities via Softmax
        probs = torch.softmax(logits, dim=-1)

        # 4. Sample from the distribution (instead of argmax)
        next_token = torch.multinomial(probs, num_samples=1)

        in_idx = torch.cat((in_idx, next_token), dim=1)

        # Stop if we hit end-of-text token
        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(in_idx.squeeze(0).tolist())