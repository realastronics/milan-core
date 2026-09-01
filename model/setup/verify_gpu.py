"""
GPU verification script — T-A2's definition of done.

Run this from inside a tmux session, after conda activate milan + install_torch.sh:

    python model/setup/verify_gpu.py

If this prints device details with no errors, T-A2 is complete.
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed. Run model/setup/install_torch.sh first.")
        return 1

    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "\nFAIL: CUDA not available. Common causes on a shared DGX:\n"
            "  - torch was installed with a CUDA version mismatched to the driver\n"
            "    (re-check `nvidia-smi` and re-run install_torch.sh with the right cuXXX tag)\n"
            "  - you're not actually on a GPU-visible node/session — check with the lab admins\n"
            "    whether GPU access needs an explicit allocation (e.g. Slurm `srun`)\n"
        )
        return 1

    device_count = torch.cuda.device_count()
    print(f"visible GPU count: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"  [{i}] {props.name} — {total_mem_gb:.1f} GB")

    # A trivial real op, not just a flag check — confirms the GPU actually computes, not just "exists"
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"\nTest matmul on GPU 0 succeeded, result shape: {tuple(y.shape)}")

    print("\nPASS — T-A2 verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
