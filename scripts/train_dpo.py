"""Use the matched preference-stage CLI for DPO."""

from pathlib import Path

if __package__:
    from .train_preference import main
else:
    from train_preference import main

if __name__ == "__main__":
    main(default_config=Path("configs/training/dpo.yaml"))
