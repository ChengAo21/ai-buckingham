"""Run the MHD MI landscape using the numerical notebook's optimization functions."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    entrypoint = ROOT / "reproduction" / "mi_landscapes" / "reproduce.py"
    namespace = runpy.run_path(str(entrypoint), run_name="mi_landscape_cli")
    namespace["main"](["--case", "mhd", *sys.argv[1:]])


if __name__ == "__main__":
    main()
