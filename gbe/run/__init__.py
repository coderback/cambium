"""gbe.run — reproducibility infra: config hashing, append-only registry, seeding,
spot-resumable checkpointing, and the RunSession context manager.

Every run in this program goes through here. Public surface:
"""

from gbe.run.config import ResolvedConfig, load_config, git_commit, git_dirty
from gbe.run.registry import append_run, REGISTRY_COLUMNS, default_registry_path
from gbe.run.seeding import seed_everything
from gbe.run.checkpoint import save_checkpoint, load_checkpoint
from gbe.run.session import RunSession

__all__ = [
    "ResolvedConfig",
    "load_config",
    "git_commit",
    "git_dirty",
    "append_run",
    "REGISTRY_COLUMNS",
    "default_registry_path",
    "seed_everything",
    "save_checkpoint",
    "load_checkpoint",
    "RunSession",
]
