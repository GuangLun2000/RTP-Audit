"""Model/tokenizer loading, prompt formatting via the tokenizer's own chat template, layer access.

Hard constraint 1 (spec §2): ``load_model`` takes exactly ONE model path. There is no
parameter for loading a base model alongside it, and none must be added.
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
    try:
        tok = AutoTokenizer.from_pretrained(model_path, **_revision_kwargs(model_path))
    except Exception as exc:  # noqa: BLE001 - re-raised with a clear message
        raise _explain_load_failure(model_path, exc) from exc
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_path: str) -> LoadedModel:
    """Load one causal LM and its tokenizer.

    CUDA: bfloat16 on ``device_map={"": 0}`` (spec §3). Without CUDA (unit/integration tests
    on a laptop) the model is loaded in float32 on CPU.
    """
    import torch
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(model_path)
    if torch.cuda.is_available():
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        device = torch.device("cuda:0")
    else:
        kwargs = {"dtype": torch.float32}
        device = torch.device("cpu")
        log.warning("CUDA not available: loading '%s' in float32 on CPU", model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs, **_revision_kwargs(model_path))
    except Exception as exc:  # noqa: BLE001
        raise _explain_load_failure(model_path, exc) from exc
    if device.type == "cpu":
        model.to(device)
    model.eval()
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
                       template_kwargs=template_kwargs)


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
