"""
Build the Kaggle notebook payload from the locally tested training module.

The notebook text is generated from kaggle_train_v2.py so the code that runs on
Kaggle is byte-identical to the code that passed the local smoke test (the only
difference is the __main__ block, which selects preflight vs. training).
"""
from __future__ import annotations

import pathlib
import re
import sys

SRC = pathlib.Path("kaggle_train_v2.py")
MAIN_BLOCK = re.compile(r'if __name__ == "__main__":\n    main\(\)\n')

TEMPLATE = '''if __name__ == "__main__":
    RUN_MODE = {mode!r}
    if RUN_MODE == "preflight":
        preflight()
    else:
        # Kaggle's image ships torch/torchvision but not onnxruntime; the ONNX
        # export + parity check needs it. Install only if missing.
        import subprocess, sys
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print("installing onnx + onnxruntime ...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "onnx", "onnxruntime"])
        main(["--epochs", "12", "--eval-every", "3"])
'''


def build(mode: str) -> str:
    src = SRC.read_text()
    if not MAIN_BLOCK.search(src):
        sys.exit("could not find __main__ block in kaggle_train_v2.py")
    return MAIN_BLOCK.sub(TEMPLATE.format(mode=mode), src)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    out = pathlib.Path(f"/tmp/opencode/notebook_{mode}.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(mode))
    compile(out.read_text(), str(out), "exec")   # syntax check
    print(f"wrote {out} ({len(out.read_text().splitlines())} lines)")
