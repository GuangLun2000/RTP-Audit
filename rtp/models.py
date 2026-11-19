"""Model/tokenizer loading, prompt formatting via the tokenizer's own chat template, layer access.

Hard constraint 1: ``load_model`` takes exactly ONE model id and loads one model. There is no parameter
for loading a second model alongside it, and none must be added. A variant id (rtp.variants) is still
one model: a registered checkpoint with a logit scale and/or one LoRA adapter applied to it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from rtp import registry

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    is_chat: bool
    device: Any
    template_kwargs: dict[str, Any] = field(default_factory=dict)  # from the registry (e.g. Qwen3 thinking off)
    logit_scale: float = 1.0  # variant "@scale=c": every output logit is multiplied by c (hook on the output head)

PROMPT_MODES = ("auto", "chat", "base")


def _is_local(model_path: str) -> bool:
    return os.path.exists(model_path)


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0", "false", "False")


def _explain_load_failure(model_path: str, exc: BaseException) -> ModelLoadError:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if any(k in lowered for k in ("gated", "401", "403", "unauthorized", "forbidden", "access to model")):
        return ModelLoadError(registry.gated_message(model_path) + f"\nOriginal error: {text}")
    return ModelLoadError(
        f"Failed to load '{model_path}'. Check the ID in rtp/registry.py, network access and HF_HOME. "
        f"No other model is substituted.\nOriginal error: {text}"
    )


def _validate_unverified(model_path: str) -> None:
    model_path = registry.base_model_id(model_path)
    spec = registry.get_model_spec(model_path)
    if spec is None or spec.verified or _is_local(model_path):
        return
    if _offline():
        log.warning("'%s' is marked 'to be confirmed' but HF_HUB_OFFLINE is set; skipping Hub validation",
                    model_path)
        return
    log.info("'%s' is marked 'to be confirmed' in the spec; validating on the Hub", model_path)
    try:
        registry.validate_repo_id(model_path, "model")
    except registry.RegistryError as exc:
        raise ModelLoadError(str(exc)) from exc


def is_cached(model_path: str) -> bool:
    """True if the checkpoint's weights are on local disk already (a local path, or a complete snapshot of the
    pinned revision in the Hugging Face cache). Used to price a load: on Colab the cache is on the VM's disk
    and is empty again in every new session."""
    base = registry.base_model_id(model_path)
    if _is_local(base):
        return True
    try:
        from pathlib import Path

        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(base, "config.json", **_revision_kwargs(model_path))
        if not isinstance(hit, str):
            return False
        snapshot = Path(hit).parent  # weight files appear here (as links) only once fully downloaded
        return any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))
    except Exception:  # noqa: BLE001 - a cache probe must never break a plan
        return False


def _revision_kwargs(model_path: str) -> dict[str, str]:
    spec = registry.get_model_spec(model_path)
    return {"revision": spec.revision} if spec is not None and spec.revision else {}


def is_chat_model(model_path: str, tokenizer: Any) -> bool:
    """Registry decides for known IDs (base checkpoints are never chat); otherwise use the tokenizer.

    The fallback is only a guess: many BASE checkpoints ship a chat template. Use
    ``--prompt-mode base|chat`` (``apply_prompt_mode``) for unregistered or local paths.
    """
    spec = registry.get_model_spec(model_path)
    if spec is not None:
        return spec.is_chat
    guess = getattr(tokenizer, "chat_template", None) is not None
    log.warning("'%s' is not in rtp/registry.py: GUESSING prompt mode '%s' from the tokenizer (base checkpoints often "
                "ship a chat template too). Pass --prompt-mode base|chat to set it explicitly.", model_path,
                "chat" if guess else "base")
    return guess


def apply_prompt_mode(lm: "LoadedModel", prompt_mode: str) -> "LoadedModel":
    """Override the chat/base decision ('auto' keeps the registry/tokenizer decision)."""
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
    if prompt_mode != "auto":
        lm.is_chat = prompt_mode == "chat"
        if lm.is_chat and getattr(lm.tokenizer, "chat_template", None) is None:
            raise ModelLoadError(f"--prompt-mode chat requested but '{lm.model_id}' has no chat template")
        log.info("prompt mode for %s set explicitly to '%s'", lm.model_id, prompt_mode)
    return lm


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    _validate_unverified(model_path)
    base = registry.base_model_id(model_path)
    try:
        tok = AutoTokenizer.from_pretrained(base, **_revision_kwargs(model_path))
    except Exception as exc:  # noqa: BLE001 - re-raised with a clear message
        raise _explain_load_failure(base, exc) from exc
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def scale_logits(model: Any, scale: float) -> Any:
    """Multiply every output logit of ``model`` by ``scale`` (a hook on the output head; returns its handle).

    The product is formed in float32: in bfloat16 a multiplication by a constant that is not a power of
    two rounds each logit differently, which would add noise to a change that must be an exact rescaling.
    The hook acts on forward passes and on ``generate`` alike.
    """
    head = model.get_output_embeddings()
    if head is None:
        raise ModelLoadError(f"{type(model).__name__} has no output embedding to scale")
    return head.register_forward_hook(lambda module, inputs, output: output.float() * scale)


def load_model(model_path: str) -> LoadedModel:
    """Load one causal LM and its tokenizer.

    CUDA: bfloat16 on ``device_map={"": 0}`` (spec §3). Without CUDA (unit/integration tests
    on a laptop) the model is loaded in float32 on CPU. For a variant id the base checkpoint is loaded
    and the variant's adapter and logit scale are applied to it.
    """
    import torch
    from transformers import AutoModelForCausalLM

    variant = registry.get_model_spec(model_path)
    base = registry.base_model_id(model_path)
    tok = load_tokenizer(model_path)
    if torch.cuda.is_available():
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        device = torch.device("cuda:0")
    else:
        kwargs = {"dtype": torch.float32}
        device = torch.device("cpu")
        log.warning("CUDA not available: loading '%s' in float32 on CPU", model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs, **_revision_kwargs(model_path))
    except Exception as exc:  # noqa: BLE001
        raise _explain_load_failure(base, exc) from exc
    if device.type == "cpu":
        model.to(device)
    if variant is not None and variant.adapter:
        from rtp import variants

        # load_adapter keeps the model's own class (no PeftModel wrapper), so layer access is unchanged
        model.load_adapter(str(variants.verified_adapter_dir(variant.adapter)))
        head = model.get_output_embeddings()
        if any("lora" in name.lower() for name, _ in head.named_modules() if name):
            # the letter readout recomputes logits from the head's own weight (rtp.measure.ladder); an adapter
            # on the head would be left out of them without any error
            raise ModelLoadError(f"adapter '{variant.adapter}' modifies the output head of {base}; train adapters "
                                 f"on the transformer blocks only (no lm_head in target_modules)")
        log.info("loaded adapter '%s' (sha256 %s) on %s", variant.adapter, (variant.adapter_sha256 or "")[:12], base)
    model.eval()
    scale = variant.logit_scale if variant is not None else 1.0
    if scale != 1.0:
        scale_logits(model, scale)
        log.info("output logits of %s are multiplied by %g", base, scale)
    chat = is_chat_model(model_path, tok)
    spec = registry.get_model_spec(model_path)
    template_kwargs = dict(spec.chat_template_kwargs) if spec is not None else {}
    if spec is None and "enable_thinking" in str(getattr(tok, "chat_template", "") or ""):
        template_kwargs = {"enable_thinking": False}
        log.warning("'%s' is unregistered and its chat template has a thinking switch: using enable_thinking=False "
                    "(register the model in rtp/registry.py to make this explicit)", model_path)
    log.info("loaded %s (%s, chat=%s, template kwargs=%s, %d blocks)", model_path, next(model.parameters()).dtype,
             chat, template_kwargs, len(get_blocks(model)))
    return LoadedModel(model=model, tokenizer=tok, model_id=model_path, is_chat=chat, device=device,
                       template_kwargs=template_kwargs, logit_scale=scale)


# --------------------------------------------------------------------------- prompts
def format_chat(tokenizer: Any, user_text: str, assistant_prefix: str | None = None,
                template_kwargs: dict[str, Any] | None = None) -> str:
    """Render a single-turn conversation with the tokenizer's built-in chat template.

    With ``assistant_prefix`` the assistant turn is pre-filled and left open
    (``continue_final_message=True``) so the model continues it. ``template_kwargs`` are
    passed to the template (e.g. ``enable_thinking=False`` for Qwen3); templates ignore
    keys they do not use.
    """
    kw = dict(template_kwargs or {})
    messages = [{"role": "user", "content": user_text}]
    if assistant_prefix is None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    messages.append({"role": "assistant", "content": assistant_prefix})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         continue_final_message=True, **kw)


def chat_prompt(lm: LoadedModel, user_text: str, assistant_prefix: str | None = None) -> str:
    """``format_chat`` with the loaded model's registry template kwargs."""
    return format_chat(lm.tokenizer, user_text, assistant_prefix, lm.template_kwargs)


def encode(tokenizer: Any, text: str, is_chat: bool) -> list[int]:
    """Tokenise a fully rendered prompt.

    Chat prompts already contain whatever special tokens the template emits, so no extra
    special tokens are added (identical to ``apply_chat_template(tokenize=True)``).
    Plain-text prompts for base models get the tokenizer's usual BOS handling.
    """
    return tokenizer(text, add_special_tokens=not is_chat)["input_ids"]


# --------------------------------------------------------------------------- layers
def get_blocks(model: Any) -> list[Any]:
    """Return the list of decoder blocks: ``model.model.layers`` first, then ``model.transformer.h``."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is not None:
        return list(layers)
    transformer = getattr(model, "transformer", None)
    h = getattr(transformer, "h", None) if transformer is not None else None
    if h is not None:
        return list(h)
    raise AttributeError(f"cannot locate decoder blocks on {type(model).__name__}: "
                         f"neither model.model.layers nor model.transformer.h exists")
