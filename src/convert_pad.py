"""Export the official Silent-Face MiniFASNet weights to ONNX.

Two checkpoints are used as an ensemble (upstream ships them together):
  2.7_80x80_MiniFASNetV2      -> models/minifasnet_v2.onnx
  4_0_0_80x80_MiniFASNetV1SE  -> models/minifasnet_v1se.onnx

IMPORTANT preprocessing note: the published reference code uses ToTensor()
(0..1), but on our data that saturates the network into a constant output for
every input. Feeding raw 0..255 BGR instead yields clean separation on a
labelled spoof set (AUC 1.000 on NUAA real vs print, after the split was fixed;
see RESEARCH_LOG). `pad.SpoofDetector` therefore does NOT divide by 255, and we
document the deviation rather than hide it.

    python convert_pad.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "third_party"))
from MiniFASNet import MiniFASNetV1SE, MiniFASNetV2  # noqa: E402

JOBS = [
    (MiniFASNetV2, "models/2.7_80x80_MiniFASNetV2.pth", "models/minifasnet_v2.onnx", (5, 5)),
    (MiniFASNetV1SE, "models/4_0_0_80x80_MiniFASNetV1SE.pth", "models/minifasnet_v1se.onnx", (5, 5)),
]


def export(cls, ckpt, out, kernel):
    model = cls(embedding_size=128, conv6_kernel=kernel, drop_p=0.2,
                num_classes=3, img_channel=3)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    rng = np.random.default_rng(0)
    probe = torch.from_numpy(rng.random((2, 3, 80, 80), dtype=np.float32) * 255)
    with torch.no_grad():
        print(f"  {Path(out).name}: probs on random input "
              f"{torch.softmax(model(probe), 1).numpy().round(3).tolist()}")

    torch.onnx.export(model, torch.zeros(1, 3, 80, 80), out, opset_version=11,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      dynamo=False)

    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = (rng.random((4, 3, 80, 80), dtype=np.float32) * 255)
    with torch.no_grad():
        a = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
    b = s.run(None, {"input": x})[0]
    b = np.exp(b - b.max(1, keepdims=True)); b = b / b.sum(1, keepdims=True)
    print(f"  {Path(out).name}: parity max|diff| = {float(np.abs(a - b).max()):.2e}")


def main():
    for cls, ckpt, out, kernel in JOBS:
        if not Path(ckpt).exists():
            print(f"missing {ckpt}")
            continue
        export(cls, ckpt, out, kernel)


if __name__ == "__main__":
    main()
