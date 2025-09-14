import argparse, math, os, copy
import torch
import torch.nn as nn
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torchao.utils import get_model_size_in_bytes
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

def set_quant_engine_for_macos():
    # Helpful on Apple Silicon; harmless elsewhere
    try:
        import platform
        if platform.system() == "Darwin":
            # If your build supports it, prefer qnnpack for ARM
            if "qnnpack" in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = "qnnpack"
    except Exception:
        pass

def load_data(calib_samples, seq_len):
    """
    Tries to pull a small split of WikiText-2 for evaluation/calibration.
    Falls back to a tiny built-in text if datasets isn't available.
    Returns a list of strings (each length ~seq_len tokens after tokenization).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"] for x in ds if x["text"].strip()]
        # Concatenate into a stream, then chunk later by tokenizer
        big_text = "\n".join(texts[:max(1000, calib_samples)])
        return [big_text], True
    except Exception:
        tiny = (
            "Language modeling measures how well a model predicts the next token. "
            "Post-training quantization compresses weights without retraining. "
            "We use dynamic INT8 for Linear layers as a simple, portable baseline."
        )
        return [tiny], False

def tokenize_stream_to_blocks(tokenizer, texts, seq_len, max_blocks):
    """Tokenize a list of large texts into up to max_blocks contiguous blocks of length seq_len."""
    ids = []
    for t in texts:
        ids.extend(tokenizer(t, return_tensors=None)["input_ids"])
    # Create blocks
    blocks = []
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        blocks.append(ids[i:i+seq_len+1])  # +1 for labels shifting
        if len(blocks) >= max_blocks:
            break
    return blocks

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, blocks, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for blk in blocks:
        input_ids = torch.tensor(blk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        labels    = torch.tensor(blk[1:],  dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(input_ids=input_ids, labels=labels)
        # outputs.loss is average over tokens
        loss = outputs.loss.item()
        total_loss += loss * labels.numel()
        total_tokens += labels.numel()
    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss)
    return mean_loss, ppl

def print_weight_dtypes(model, prefix=""):
    print(f"\n[{prefix}] weight dtypes for modules with weights:")
    for name, mod in model.named_modules():
        # Quantized Linear exposes weight() method; float Linear has .weight Parameter
        w = None
        if hasattr(mod, "weight"):
            try:
                # Try quantized accessor first
                w = mod.weight() if callable(getattr(mod, "weight")) else mod.weight
            except Exception:
                pass
        if w is not None:
            try:
                print(f" - {name:40s} ({type(mod).__name__:15s}) -> {str(w.dtype)}")
            except Exception:
                print(f" - {name:40s} ({type(mod).__name__:15s}) -> (dtype unavailable)")

def main():
    parser = argparse.ArgumentParser(description="Post-Training Quantization (INT8 dynamic) for a small LLM")
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="HF model id (e.g., distilgpt2, gpt2, sshleifer/tiny-gpt2)")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--calib_samples", type=int, default=200, help="Number of blocks for eval/calib")
    parser.add_argument("--save_model", type=bool, default=False, help="Save the model to the models/ptq_llm directory")
    parser.add_argument("--detailed", type=bool, default=False, help="Detailed breakdown of layer types")
    args = parser.parse_args()

    set_quant_engine_for_macos()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cpu")
    model_fp32 = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model_fp32.eval()

    texts, used_ds = load_data(args.calib_samples, args.seq_len)
    blocks = tokenize_stream_to_blocks(tokenizer, texts, args.seq_len, args.calib_samples)

    base_loss, base_ppl = evaluate_perplexity(model_fp32, tokenizer, blocks, device)

    model_int8 = copy.deepcopy(model_fp32)
    quantize_(
        model_int8, Int8WeightOnlyConfig()
    )
    q_loss, q_ppl = evaluate_perplexity(model_int8, tokenizer, blocks, device)

    if args.save_model:
        os.makedirs("models/ptq_llm", exist_ok=True)
        torch.save(model_fp32.state_dict(), f"models/ptq_llm/{args.model_name}_fp32.pt")
        torch.save(model_int8.state_dict(), f"models/ptq_llm/{args.model_name}_int8.pt")

    print("\n-------BASELINE METRICS--------")
    print(f"Baseline (FP32)  - loss: {base_loss:.4f}, perplexity: {base_ppl:.2f}")
    print(f"FP32 SIZE: {get_model_size_in_bytes(model_fp32) / 1e6:.2f} MB")
    print("--------------------------------\n")
    print("-------QUANTIZED METRICS---------")
    print(f"Quantized (INT8) - loss: {q_loss:.4f}, perplexity: {q_ppl:.2f}")
    print(f"INT8 SIZE: {get_model_size_in_bytes(model_int8) / 1e6:.2f} MB")
    print("--------------------------------\n")

    if args.detailed:
        print("--------------------------------")
        print("DETAILED BREAKDOWN OF LAYER TYPES")
        print("--------------------------------")
        print(model_fp32)
        print(model_int8)
        print("--------------------------------")

if __name__ == "__main__":
    main()