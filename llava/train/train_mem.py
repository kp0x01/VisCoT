# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import os

# Need to call this before importing transformers unless flash attention is disabled.
if os.environ.get("LLAVA_DISABLE_FLASH_ATTN", "0") not in ("1", "true", "True"):
    try:
        from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()
    except ImportError:
        print("[WARN] flash-attn not available, falling back to standard attention.")
else:
    print("[INFO] LLAVA_DISABLE_FLASH_ATTN set, skipping flash attention monkey patch.")

from llava.train.train import train

if __name__ == "__main__":
    train()
