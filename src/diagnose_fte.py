"""
Diagnose failure-to-enroll (FTE): why do some probes yield no usable face?

For every probe image, record image size and what the SCRFD detector returns
under several settings. Used to decide the final detection policy.

Output: data/fte_diagnosis.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PROBE = Path("data/pak_fig/probe")


def run_detector(app, img):
    return app.get(img)


def main():
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    files = sorted(p for p in PROBE.rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"probes: {len(files)}", flush=True)

    rows = []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            rows.append({"file": str(p.relative_to(PROBE)), "read": False})
            continue
        h, w = img.shape[:2]
        faces = app.get(img)
        best = max((f.det_score for f in faces), default=0.0)
        row = {
            "file": str(p.relative_to(PROBE)),
            "w": w, "h": h,
            "n_faces": len(faces),
            "best_score": float(best),
            "face_w": float(max((f.bbox[2] - f.bbox[0] for f in faces), default=0.0)),
            "face_h": float(max((f.bbox[3] - f.bbox[1] for f in faces), default=0.0)),
            "iod": float(max((np.linalg.norm(f.kps[0] - f.kps[1]) for f in faces), default=0.0)),
            # retry small/failed images upscaled 2x
            "n_faces_up2": None,
            "best_score_up2": None,
        }
        if len(faces) == 0 or row["iod"] < 40:
            up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            f2 = app.get(up)
            row["n_faces_up2"] = len(f2)
            row["best_score_up2"] = float(max((f.det_score for f in f2), default=0.0))
            row["iod_up2"] = float(max((np.linalg.norm(f.kps[0] - f.kps[1]) for f in f2),
                                       default=0.0))
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

    out = Path("data/fte_diagnosis.json")
    out.write_text(json.dumps(rows, indent=1))

    no_face = [r for r in rows if r.get("n_faces", 0) == 0]
    low_iod = [r for r in rows if 0 < r.get("iod", 0) < 64]
    fixed_by_up2 = [r for r in no_face if (r.get("n_faces_up2") or 0) > 0]
    print(f"\nno face @640/0.5: {len(no_face)}/{len(rows)}")
    print(f"iod<64        : {len(low_iod)}")
    print(f"rescued by 2x upscale: {len(fixed_by_up2)}")
    if no_face:
        sizes = np.array([[r["w"], r["h"]] for r in no_face])
        print(f"no-face image size: median {np.median(sizes,0)} "
              f"min {sizes.min(0)} max {sizes.max(0)}")
        print("examples:", [r["file"] for r in no_face[:8]])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
