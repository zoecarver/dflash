"""CPU-only speculative decoding: ground truth acceptance rate.

Runs DFlashDraftModel.spec_generate on CPU with the real Qwen3 target model.
This gives us the "ceiling" acceptance rate with perfect precision.
"""
import sys
sys.path.insert(0, "/tmp")

import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from safetensors.torch import load_file
from dflash_ref import DFlashDraftModel
from utils_minimal import extract_context_feature, sample

TARGET_DIR = "/workspace/qwen-coder-30b-a3b/weights"
DRAFT_DIR = "/workspace/qwen-coder-30b-a3b/dflash"


def main():
    device = torch.device("cpu")

    # Load target model
    print("Loading target model...")
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR,
        torch_dtype=torch.bfloat16,
    )
    target.eval()
    print(f"  Target loaded in {time.time()-t0:.1f}s")

    # Load draft model
    print("Loading draft model...")
    t0 = time.time()
    with open(f"{DRAFT_DIR}/config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen3Config(**cfg_dict)
    config._attn_implementation = "eager"
    draft = DFlashDraftModel(config)
    state_dict = load_file(f"{DRAFT_DIR}/model.safetensors")
    draft.load_state_dict(state_dict)
    draft = draft.to(torch.bfloat16)
    draft.eval()
    print(f"  Draft loaded in {time.time()-t0:.1f}s")

    # Tokenize
    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    prompt = "Write a Python function that computes fibonacci numbers."
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
    input_ids = tok(text, return_tensors="pt")["input_ids"]
    print(f"Prompt: {input_ids.shape[1]} tokens")

    # Patch spec_generate to print per-step acceptance
    original_spec = draft.spec_generate

    @torch.inference_mode()
    def verbose_spec_generate(target_model, input_ids, max_new_tokens, stop_token_ids, temperature):
        draft.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = draft.block_size

        from transformers import DynamicCache
        output_ids = torch.full(
            (1, max_length + block_size),
            draft.mask_token_id,
            dtype=torch.long,
            device=target_model.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target_model.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill
        print("Prefill...")
        t0 = time.time()
        output = target_model(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        print(f"  Prefill: {time.time()-t0:.1f}s")
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

        # Decode
        acceptance_lengths = []
        start = num_input_tokens
        step = 0
        print("Decoding...")
        while start < max_length:
            ts = time.time()
            block_output_ids = output_ids[:, start : start + block_size].clone()
            noise_embedding = target_model.model.embed_tokens(block_output_ids)
            draft_output = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            draft_logits = target_model.lm_head(draft_output[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target_model(
                block_output_ids,
                position_ids=position_ids[:, start : start + block_size],
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(output.logits, temperature)
            acc_len = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acc_len + 1] = block_output_ids[:, : acc_len + 1]
            output_ids[:, start + acc_len + 1] = posterior[:, acc_len]
            start += acc_len + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)[:, :acc_len + 1, :]

            step += 1
            acceptance_lengths.append(acc_len + 1)
            avg = sum(acceptance_lengths) / len(acceptance_lengths)
            el = time.time() - ts
            gen = sum(acceptance_lengths)
            print(f"  step {step}: acc={acc_len+1}/{block_size} avg={avg:.1f} {el:.1f}s gen={gen}")

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != draft.mask_token_id]
        return output_ids

    # Run
    print("\n=== Speculative Decoding (CPU, bf16) ===")
    out = verbose_spec_generate(
        target, input_ids, max_new_tokens=64,
        stop_token_ids=[151643, 151645], temperature=0.0,
    )
    print(f"\n--- Output ---\n{tok.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
