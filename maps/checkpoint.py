"""
Checkpoint loading helpers.

Shipped MAPS checkpoints store EMA weights under
ckpt['ema_state_dict']['shadow'] (the paper reports EMA-weight inference);
baseline checkpoints store raw weights under 'model_state_dict'. A bare
``ck.get('model_state_dict', ck.get('ema_state_dict', ck))`` followed by
``load_state_dict(strict=False)`` matches NO keys on the EMA-only layout
and silently leaves the model randomly initialized. These helpers resolve
the layout explicitly and fail loudly on any key mismatch.

Functions:
    extract_model_state -- resolve a checkpoint object to a raw state_dict
    load_state_checked  -- load a state_dict, raising on any key mismatch
"""


def extract_model_state(ckpt, prefer_ema=True):
    """Return the raw model state_dict contained in a checkpoint object.

    Resolution order with prefer_ema=True (paper protocol: EMA weights
    used for all reported inference):
        1. ckpt['ema_state_dict']['shadow']  (EMA wrapper dict)
        2. ckpt['ema_state_dict']            (plain EMA state_dict)
        3. ckpt['model_state_dict']
        4. ckpt['state_dict']
        5. ckpt itself (checkpoint IS a state_dict)
    With prefer_ema=False the EMA entry is tried after 'model_state_dict'
    and 'state_dict' (for baselines saved without EMA).
    """
    if not isinstance(ckpt, dict):
        return ckpt

    def _ema():
        ema = ckpt.get('ema_state_dict')
        if isinstance(ema, dict) and 'shadow' in ema:
            return ema['shadow']
        return ema

    if prefer_ema and 'ema_state_dict' in ckpt:
        return _ema()
    for key in ('model_state_dict', 'state_dict'):
        if key in ckpt:
            return ckpt[key]
    if 'ema_state_dict' in ckpt:
        return _ema()
    return ckpt


def load_state_checked(model, state, *, strict=True, label='checkpoint'):
    """Load `state` into `model`, raising RuntimeError on any key mismatch.

    Unlike bare ``load_state_dict(strict=False)``, missing/unexpected keys
    are never swallowed: they raise even when strict=False (the flag is
    kept for signature compatibility only), so an incompatible checkpoint
    can never silently leave the model at its random initialization.
    """
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        lines = [f"{label}: state_dict does not match "
                 f"{type(model).__name__}"]
        if missing:
            lines.append(f"  missing keys ({len(missing)}): "
                         f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            lines.append(f"  unexpected keys ({len(unexpected)}): "
                         f"{unexpected[:10]}"
                         f"{' ...' if len(unexpected) > 10 else ''}")
        lines.append("  hint: pass the checkpoint through "
                     "extract_model_state() first")
        raise RuntimeError('\n'.join(lines))
    return result
