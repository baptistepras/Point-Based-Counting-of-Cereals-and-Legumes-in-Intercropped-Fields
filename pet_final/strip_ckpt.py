"""Strip a PET ShanghaiTech checkpoint down to model weights only.

Keeping just {'model': state_dict} makes PET's main.py load the weights and start
fine-tuning from epoch 0 (the optimizer/lr_scheduler/epoch keys are intentionally
dropped so main.py's resume branch does not restore training state).
"""

import argparse
from pathlib import Path
import torch


def main() -> None:
    """Read a full PET checkpoint and rewrite it as {'model': state_dict}."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="downloaded SHA checkpoint (.pth)")
    parser.add_argument(
        "--dst",
        default=str(Path(__file__).resolve().parent / "pretrained" / "pet_sha_model_only.pth"),
        help="output path for the weights-only checkpoint",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, dst)
    print(f"wrote {dst} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
