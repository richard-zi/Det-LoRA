"""Variant-specific training presets."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

SUPPORTED_VARIANTS = ("nano", "small", "base", "medium", "large")


L40_FINAL_PRESET: Dict[str, Dict[str, float | int]] = {
    "nano": {
        "batch_size": 16,
        "lr": 1.5e-4,
        "weight_decay": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": 30,
        "metrics_eval_every": 2,
    },
    "small": {
        "batch_size": 12,
        "lr": 1.25e-4,
        "weight_decay": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": 30,
        "metrics_eval_every": 2,
    },
    "base": {
        "batch_size": 8,
        "lr": 1.0e-4,
        "weight_decay": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": 30,
        "metrics_eval_every": 2,
    },
    "medium": {
        "batch_size": 6,
        "lr": 1.0e-4,
        "weight_decay": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": 30,
        "metrics_eval_every": 2,
    },
    "large": {
        "batch_size": 4,
        "lr": 8.0e-5,
        "weight_decay": 1e-4,
        "lora_rank": 12,
        "lora_alpha": 24,
        "epochs": 30,
        "metrics_eval_every": 2,
    },
}


PRESET_PROFILES: Dict[str, Dict[str, Dict[str, float | int]]] = {
    "l40_final": L40_FINAL_PRESET,
}


def expand_model_variants(models: Iterable[str]) -> list[str]:
    """Expand 'all' into all supported variants while preserving order."""
    expanded: list[str] = []
    seen = set()
    for model in models:
        candidates = SUPPORTED_VARIANTS if model == "all" else (model,)
        for candidate in candidates:
            if candidate not in SUPPORTED_VARIANTS:
                raise ValueError(
                    f"Unsupported model variant '{candidate}'. "
                    f"Expected one of {', '.join(SUPPORTED_VARIANTS)} or 'all'."
                )
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return expanded


def resolve_variant_settings(
    *,
    variant: str,
    preset_name: Optional[str],
    base_defaults: Dict[str, float | int],
    overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, float | int]:
    """Resolve CLI/config values with precedence: overrides > preset > defaults."""
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported model variant '{variant}'")

    resolved = dict(base_defaults)
    if preset_name:
        profile = PRESET_PROFILES.get(preset_name)
        if profile is None:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {', '.join(sorted(PRESET_PROFILES))}"
            )
        resolved.update(profile[variant])

    for key, value in (overrides or {}).items():
        if value is not None:
            resolved[key] = value
    return resolved
