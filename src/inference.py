"""
End-to-end inference pipeline: detect -> align -> quality gate -> embed -> match.
CPU-friendly. Uses InsightFace (buffalo_l) by default.
"""
import os
import sys
import json
import argparse
import numpy as np

from imgio import imread_any
import cv2
from pathlib import Path

# InsightFace
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------- Configuration ----------
DEFAULT_MODEL = "buffalo_l"
DET_SIZE = (640, 640)
DET_PAD_RETRY = 0.3   # reflected border fraction when no face is found
EMB_DIM = 512
EPSILON = 1e-12  # small constant to prevent division by zero in normalization
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _asset(name: str) -> str:
    """Locate a weight file whether the tree is the repo or the submission zip.

    Searches multiple locations and raises FileNotFoundError with all attempted
    paths if the asset cannot be found anywhere.
    """
    search_locations = (_ROOT / "models", _ROOT / "submit" / "models", _HERE, _HERE / "models")
    for base in search_locations:
        p = base / name
        if p.exists():
            return str(p)
    # Asset not found; raise with helpful error message listing all searched locations
    locations_str = "\n  ".join(str(base / name) for base in search_locations)
    raise FileNotFoundError(
        f"Model weight file '{name}' not found. Searched:\n  {locations_str}"
    )


DEFAULT_PAD_MODELS = [_asset("minifasnet_v2.onnx"), _asset("minifasnet_v1se.onnx")]
PAD_THRESHOLD = 0.02  # 2.3% BPCER on our probes / 1.3% APCER on NUAA


class FacePipeline:
    def __init__(self, model_name=DEFAULT_MODEL, det_size=DET_SIZE, ctx_id=-1,
                 pad_models="default", pad_threshold=PAD_THRESHOLD,
                 onnx_path=None, models_root="~/.insightface"):
        self.models_root = models_root
        self.model_name = model_name
        if onnx_path:
            # detection still comes from the insightface pack; the recogniser is
            # an external 112x112 -> 512 ONNX model (e.g. AdaFace)
            import onnxruntime as ort
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                    root=models_root)
            so = ort.SessionOptions()
            so.intra_op_num_threads = os.cpu_count() or 4
            self.sess = ort.InferenceSession(str(onnx_path), so,
                                             providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
        else:
            self.sess = None
            self.app = FaceAnalysis(name=model_name,
                                    allowed_modules=["detection", "recognition"],
                                    root=models_root)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_size = det_size
        self._lo = None
        # liveness gate: a spoofed input must never reach the matcher
        self.spoof = None
        if pad_models:
            from pad import SpoofDetector
            pm = DEFAULT_PAD_MODELS if pad_models == "default" else pad_models
            self.spoof = SpoofDetector(pm, threshold=pad_threshold)

    # ---------- Quality gate ----------
    @staticmethod
    def quality_ok(face, img, min_iod=64, max_yaw=30, min_blur=40):
        """Cheap quality filters. Returns (ok, reason)."""
        # Face size (IOD)
        kps = face.kps  # 5 landmarks
        iod = np.linalg.norm(kps[0] - kps[1])
        if iod < min_iod:
            return False, f"low_iod_{iod:.0f}"

        # Blur (Laplacian variance on face crop)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if (y2 - y1) > 0 and (x2 - x1) > 0:
            gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if gray is not None and gray.size > 0:
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur < min_blur:
                    return False, f"blur_{blur:.0f}"

        # Pose (yaw from 5-pt landmarks, rough estimate)
        # Use nose position relative to eye midpoint
        eye_mid = (kps[0] + kps[1]) / 2
        nose = kps[2]
        dx = nose[0] - eye_mid[0]
        # Normalize by IOD
        yaw_approx = abs(dx) / (iod + EPSILON) * 90  # rough degrees
        if yaw_approx > max_yaw:
            return False, f"yaw_{yaw_approx:.0f}"

        return True, "ok"

    # ---------- Detection + embedding ----------
    @staticmethod
    def _area(f):
        return float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def _iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _detect(self, img, det_thresh=0.5, trace=None):
        """Two escalations, both for SCRFD's scene-context bias:
        nothing found -> reflected border; only tiny faces -> 320 pass.

        `trace` is an optional list that the demo UI appends human-readable
        stage notes to. It never changes the result.
        """
        notes = trace if trace is not None else []
        h, w = img.shape[:2]
        faces = [f for f in self.app.get(img) if f.det_score >= det_thresh]
        notes.append(f"pass 1 @640 on {w}x{h}: {len(faces)} face(s)")
        if not faces:
            b = int(round(max(img.shape[:2]) * DET_PAD_RETRY))
            if b > 0:
                padded = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REFLECT_101)
                f2 = [f for f in self.app.get(padded) if f.det_score >= det_thresh]
                if f2:
                    notes.append(f"escalation 1: nothing found -> reflected border "
                                 f"+{b}px -> {len(f2)} face(s)")
                    return padded, f2
                notes.append(f"escalation 1: reflected border +{b}px -> still nothing")
        if faces:
            big = max(faces, key=self._area)
            iod = float(np.linalg.norm(big.kps[0] - big.kps[1]))
            frac = self._area(big) / max(1, img.shape[0] * img.shape[1])
            if iod < 64 or frac < 0.02:
                if self._lo is None:
                    self._lo = FaceAnalysis(name=self.model_name if not self.sess
                                            else "buffalo_l",
                                            allowed_modules=["detection"],
                                            root=self.models_root)
                    self._lo.prepare(ctx_id=-1, det_size=(320, 320))
                extra = [f for f in self._lo.get(img)
                         if f.det_score >= det_thresh
                         and all(self._iou(f.bbox, g.bbox) < 0.5 for g in faces)]
                faces = faces + extra
                notes.append(f"escalation 2: largest IOD {iod:.0f}px, area "
                             f"{frac*100:.1f}% -> 320px pass -> +{len(extra)} face(s)")
            else:
                notes.append(f"no escalation: largest IOD {iod:.0f}px, "
                             f"area {frac*100:.1f}%")
        return img, faces

    def _embed(self, img, face):
        if self.sess is None:
            emb = face.normed_embedding  # already L2-normalised by InsightFace
            if emb is None:
                # A face from the 320px escalation comes from a detection-only
                # FaceAnalysis, so it carries no embedding. Recompute it from
                # the aligned crop with the pack's recogniser.
                rec = self.app.models.get("recognition")
                if rec is not None:
                    crop = face_align.norm_crop(img, landmark=face.kps, image_size=112)
                    emb = rec.get_feat(crop).flatten()
                else:
                    emb = face.embedding
                if emb is None:
                    raise RuntimeError("no recogniser available to embed this face")
                emb = emb / (np.linalg.norm(emb) + EPSILON)
            return emb
        crop = face_align.norm_crop(img, landmark=face.kps, image_size=112)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input_name: x})[0][0]
        return out / (np.linalg.norm(out) + EPSILON)

    def get_faces(self, img, quality_filter=True, gate_pad=True, trace=None):
        """Returns list of (face_obj, embedding, quality_reason).

        `gate_pad=False` is for gallery enrolment: an ID-card scan or a printed
        portrait is a legitimate enrolment source, so liveness must not be
        applied there. It is always applied to probes.
        """
        img, faces = self._detect(img, trace=trace)
        results = []
        for f in faces:
            if self.spoof is not None and gate_pad:
                pv = self.spoof.predict(img, f.bbox)
                if not pv["is_live"]:
                    # a liveness rejection is final: no embedding is computed
                    results.append((f, None, f"spoof_{pv['live_prob']:.2f}", None))
                    continue
            if quality_filter:
                ok, reason = self.quality_ok(f, img)
                if not ok:
                    # keep the embedding for salvage: a soft face is better than
                    # no enrolment, and with an external ONNX recogniser there is
                    # no face.embedding to fall back on.
                    results.append((f, None, reason, self._embed(img, f)))
                    continue
            emb = self._embed(img, f)
            results.append((f, emb, "ok", emb))
        return results

    def embed_image(self, img_path, quality_filter=True, gate_pad=True):
        """Extract best embedding from an image (largest face passing quality)."""
        img = imread_any(img_path)
        if img is None:
            return None, "read_error"
        faces = self.get_faces(img, quality_filter, gate_pad)
        # Pick largest face that passed quality
        good = [(f, e) for f, e, r, _ in faces if e is not None]
        if not good:
            # Quality failures are rescued (we would rather embed a soft face
            # than fail to enrol), but a liveness rejection is final: reviving
            # it here would make the anti-spoofing gate decorative.
            salvage = [t for t in faces if not t[2].startswith("spoof_")]
            if salvage:
                f = max(salvage, key=lambda x: (x[0].bbox[2]-x[0].bbox[0])*(x[0].bbox[3]-x[0].bbox[1]))
                e = f[3] if f[3] is not None else f[1]
                if e is not None:
                    return e, "fallback_quality" if f[2] != "ok" else "ok"
            if faces:
                return None, faces[0][2]
            return None, "no_face"
        # Largest good face
        best = max(good, key=lambda x: (x[0].bbox[2]-x[0].bbox[0])*(x[0].bbox[3]-x[0].bbox[1]))
        return best[1], "ok"

    # ---------- Matching ----------
    @staticmethod
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPSILON))

    def match(self, probe_emb, gallery_embs, gallery_ids, threshold=0.4):
        """1:N match. Returns (best_id, best_score, is_match)."""
        if probe_emb is None or len(gallery_embs) == 0:
            return None, 0.0, False
        gallery = np.array(gallery_embs)
        gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + EPSILON)
        probe = probe_emb / (np.linalg.norm(probe_emb) + EPSILON)
        scores = gallery @ probe
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_id = gallery_ids[best_idx]
        return best_id, best_score, best_score >= threshold


def build_gallery(pipeline, gallery_dir, quality_filter=True):
    """Build gallery embeddings from a directory of images.
    Expected structure: gallery_dir/<identity>/<image>.jpg
    """
    gallery_dir = Path(gallery_dir)
    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory not found or not a directory: {gallery_dir}")
    embs, ids = [], []
    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        pid = person_dir.name
        person_embs = []
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            emb, reason = pipeline.embed_image(img_path, quality_filter, gate_pad=False)
            if emb is not None:
                person_embs.append(emb)
        if person_embs:
            # Quality-weighted average (simple: unweighted mean of normalized embs)
            mean_emb = np.mean(person_embs, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + EPSILON)
            embs.append(mean_emb)
            ids.append(pid)
    return embs, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", required=True, help="Gallery dir: gallery/<id>/<img>")
    parser.add_argument("--probe", required=True, help="Probe image or dir")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["buffalo_l", "antelopev2"],
                        help="insightface model pack (antelopev2 = R100/Glint360K)")
    parser.add_argument("--no-pad", action="store_true",
                        help="disable the anti-spoofing gate")
    parser.add_argument("--pad-threshold", type=float, default=PAD_THRESHOLD)
    args = parser.parse_args()

    pipeline = FacePipeline(model_name=args.model,
                            pad_models=None if args.no_pad else "default",
                            pad_threshold=args.pad_threshold)

    # Build gallery
    print("Building gallery...")
    gallery_embs, gallery_ids = build_gallery(pipeline, args.gallery)
    print(f"Gallery: {len(gallery_ids)} identities")

    # Process probes
    probe_path = Path(args.probe)
    probe_files = [probe_path] if probe_path.is_file() else sorted(
        p for p in probe_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    results = []
    for pf in probe_files:
        if pf.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        emb, reason = pipeline.embed_image(pf, quality_filter=True)
        best_id, score, is_match = pipeline.match(emb, gallery_embs, gallery_ids, args.threshold)
        key = pf.name if probe_path.is_file() else str(pf.relative_to(probe_path))
        results.append({
            "probe": key,
            "predicted_id": best_id if is_match else "unknown",
            "score": round(score, 4),
            "quality": reason,
            "match": is_match,
        })
        print(f"{key}: {best_id if is_match else 'unknown'} (score={score:.4f}, q={reason})")

    # Save CSV
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["probe", "predicted_id", "score", "quality", "match"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()