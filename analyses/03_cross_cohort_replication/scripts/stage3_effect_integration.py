from __future__ import annotations


def build_effect_integration_preview() -> None:
    """Future deterministic merge of already-computed effect tables.

    This placeholder must not fit models or run new differential-abundance
    regression. The setup pass already wrote a preview from existing outputs.
    """
    raise RuntimeError("Effect integration execution is disabled in Stage 3 setup. Use --allow-heavy only after GO criteria are approved.")

