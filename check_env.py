"""
Step 0 — Environment Setup
===========================
Before building anything, confirm PyTorch is installed and check whether
you have a GPU available. You do NOT need a GPU for this project — every
module here is small enough to run on CPU. A GPU only matters later if
you try training on a real translation dataset (step 12+).

Run this file directly:
    python check_env.py
"""

import sys


def main():
    print(f"Python version: {sys.version}")

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it with:")
        print("    pip install torch --break-system-packages")
        print("(or without the flag, inside a virtual environment)")
        return

    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA available: YES ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA available: NO — running on CPU is fine for steps 0-11.")

    # Quick sanity check: create a tensor and do a matmul, the core op
    # behind every attention computation you'll write from here on.
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)
    c = a @ b
    assert c.shape == (2, 4)
    print("Basic matmul sanity check passed. c.shape =", tuple(c.shape))
    print("\nEnvironment OK. Move on to step1_attention_basics/")


if __name__ == "__main__":
    main()
