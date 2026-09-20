"""Weights & Biases logging: optional, and structurally unable to stop a run.

A long training run must not die because a metrics service is unreachable, a
key expired, or the machine lost DNS.  Every call here is wrapped, and the
first failure disables logging for the rest of the process rather than raising.
The stdout log remains the primary record.

Credentials come from `.env` (gitignored) or the ambient environment.  The value
is moved into `os.environ` for the `wandb` client and is never read back, logged
or printed by anything here.
"""
from __future__ import annotations

import os
from pathlib import Path

_run = None
_off = False          # set on the first failure; never retried


def load_env(path: str | Path = ".env") -> bool:
    """Load KEY=VALUE lines into os.environ.  Existing values win.

    Deliberately minimal -- no dependency, no interpolation, no export syntax.
    Values are never echoed: this returns only whether a key is now present.
    """
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return bool(os.environ.get("WANDB_API_KEY"))


def init(project: str, name: str, config: dict, mode: str = "auto",
         run_id: str | None = None) -> bool:
    """Start a run, or resume `run_id`.  Returns whether logging is live.

    `mode='auto'` uses W&B when a key is present and does nothing when it is
    not, so the same command works with and without credentials.  Passing the
    id stored in a checkpoint continues the same chart across a restart instead
    of starting a disconnected second one.
    """
    global _run, _off
    if mode == "off":
        _off = True
        return False
    have_key = load_env()
    if not have_key and mode == "auto":
        print("wandb: no WANDB_API_KEY, logging to stdout only", flush=True)
        _off = True
        return False
    try:
        import wandb
        _run = wandb.init(project=project, name=name, config=config,
                          id=run_id, resume="allow" if run_id else None,
                          settings=wandb.Settings(init_timeout=60))
        print(f"wandb: {getattr(_run, 'url', '(offline)')}", flush=True)
        return True
    except Exception as e:                       # noqa: BLE001 - never fatal
        print(f"wandb: disabled ({type(e).__name__}); training continues",
              flush=True)
        _off = True
        return False


def run_id() -> str | None:
    """The id to checkpoint, so a resume continues this run rather than fork."""
    return getattr(_run, "id", None) if _run is not None else None


def log(metrics: dict, step: int | None = None) -> None:
    global _off
    if _run is None or _off:
        return
    try:
        _run.log({k: v for k, v in metrics.items() if v == v}, step=step)
    except Exception as e:                       # noqa: BLE001
        print(f"wandb: log failed ({type(e).__name__}); disabling", flush=True)
        _off = True


def finish() -> None:
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception:                            # noqa: BLE001
        pass
    _run = None
