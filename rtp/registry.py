"""Single source of truth for every model and dataset ID used by the project.

No other module may hard-code a Hugging Face repo ID. IDs that the spec marks as
"to be confirmed" carry ``verified=False`` and are checked against the Hub at
runtime (``validate_repo_id``); a failed check raises instead of substituting
another resource.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str  # "qwen", "tulu3", "olmo3", "llama"
    stage: str  # "base", "sft", "dpo", "final", "instruct"
    role: str  # "smoke", "base", "stage", "generality"
    is_chat: bool  # False -> measured with plain text continuation, never a chat template
    gated: bool  # True -> licence must be accepted on the Hub and HF_TOKEN set
    verified: bool  # False -> spec marks the ID "to be confirmed"; validated at runtime
    note: str = ""
    # extra keyword arguments for tokenizer.apply_chat_template, as (key, value) pairs
    chat_template_kwargs: tuple[tuple[str, object], ...] = ()
    # optional Hub commit to pin (None = default branch); the loaded commit is recorded per stage either way
    revision: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    role: str
    verified: bool
    note: str = ""


SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_MODELS = [
    ModelSpec(SMOKE_MODEL, "qwen", "instruct", "smoke", True, False, True, "development and smoke tests"),
    # Tulu 3 (main platform)
    ModelSpec("meta-llama/Llama-3.1-8B", "tulu3", "base", "base", False, True, True,
              "Tulu 3 base; not a chat model; gated (accept the Llama 3.1 licence)"),
    ModelSpec("allenai/Llama-3.1-Tulu-3-8B-SFT", "tulu3", "sft", "stage", True, False, True),
    ModelSpec("allenai/Llama-3.1-Tulu-3-8B-DPO", "tulu3", "dpo", "stage", True, False, True),
    ModelSpec("allenai/Llama-3.1-Tulu-3-8B", "tulu3", "final", "stage", True, False, True),
    # OLMo 3 (replication)
    ModelSpec("allenai/Olmo-3-1025-7B", "olmo3", "base", "base", False, False, False,
              "OLMo 3 base; spec marks this ID as to be confirmed"),
    ModelSpec("allenai/Olmo-3-7B-Instruct-SFT", "olmo3", "sft", "stage", True, False, True),
    ModelSpec("allenai/Olmo-3-7B-Instruct-DPO", "olmo3", "dpo", "stage", True, False, True),
    ModelSpec("allenai/Olmo-3-7B-Instruct", "olmo3", "final", "stage", True, False, True),
    # Generality (detection only)
    # Qwen3's template enables thinking by default: the next token would be "<think>", never a letter.
    ModelSpec("Qwen/Qwen3-8B", "qwen", "final", "generality", True, False, True,
              "thinking disabled via the template switch", chat_template_kwargs=(("enable_thinking", False),)),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen", "final", "generality", True, False, True),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama", "final", "generality", True, True, True, "gated"),
]

_DATASETS = [
    DatasetSpec("allenai/llama-3.1-tulu-3-8b-preference-mixture", "tulu3_dpo_carrier", True,
                "271,409 preference pairs; M5 carrier data"),
    DatasetSpec("allenai/dolci-3-instruct-dpo-with-metadata", "olmo3_dpo_carrier", False,
                "spec marks this ID as to be confirmed; M5"),
]

MODELS: dict[str, ModelSpec] = {m.model_id: m for m in _MODELS}
DATASETS: dict[str, DatasetSpec] = {d.dataset_id: d for d in _DATASETS}

# Non-Hub data sources (spec §5); downloaded manually, parsed from raw files.
AMAZON_REVIEWS_2023_URL = "https://amazon-reviews-2023.github.io"


def get_model_spec(model_id: str) -> ModelSpec | None:
    """Return the registry entry for ``model_id`` or None for unregistered paths (e.g. local adapters)."""
    return MODELS.get(model_id)


def get_dataset_spec(dataset_id: str) -> DatasetSpec | None:
    return DATASETS.get(dataset_id)


class RegistryError(RuntimeError):
    pass


def validate_repo_id(repo_id: str, repo_type: str = "model") -> None:
    """Check that ``repo_id`` exists on the Hub under exactly that name.

    Raises RegistryError when the repo is missing, gated without access, or the Hub
    resolves it to a *different* repo (a rename). We never follow a rename silently.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.model_info(repo_id) if repo_type == "model" else api.dataset_info(repo_id)
    except GatedRepoError as exc:
        raise RegistryError(gated_message(repo_id)) from exc
    except RepositoryNotFoundError as exc:
        raise RegistryError(
            f"{repo_type} '{repo_id}' was not found on the Hugging Face Hub. The spec marks some IDs as "
            f"'to be confirmed'; fix the ID in rtp/registry.py after checking the Hub. Not substituting."
        ) from exc
    resolved = getattr(info, "id", repo_id)
    if resolved and resolved.lower() != repo_id.lower():
        raise RegistryError(
            f"{repo_type} '{repo_id}' resolves to '{resolved}' on the Hub (renamed?). Confirm the intended "
            f"resource and update rtp/registry.py explicitly. Not substituting."
        )


def gated_message(repo_id: str) -> str:
    return (
        f"Cannot access '{repo_id}'. It is a gated repository: open https://huggingface.co/{repo_id}, "
        f"accept the licence with your account, then expose a read token as the HF_TOKEN environment "
        f"variable (Colab: Secrets -> HF_TOKEN). No other model is substituted."
    )
