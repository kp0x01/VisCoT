from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

try:
    from .language_model.llava_llamantk import LlavaLlamaNTKForCausalLM, LlavaNTKConfig
except Exception as exc:  # Skip NTK variant when flash-attn is unavailable
    LlavaLlamaNTKForCausalLM = None  # type: ignore
    LlavaNTKConfig = None  # type: ignore
    print(f"[WARN] Failed to import LlavaLlamaNTK variant: {exc}")

# from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
