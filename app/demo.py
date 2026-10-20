#!/usr/bin/env python3
"""Presentation UI for the TechFest face-recognition system.

    uv run python app/demo.py                 # http://127.0.0.1:7860
    uv run python app/demo.py --share         # public link, for a remote projector

Runs offline on CPU against the same weights the submission ships. The gallery
is the 23 public-figure identities from REPORT.md, enrolled from
data/pak_fig/gallery; embeddings are cached in app/.cache after the first build
(~3 minutes), so later starts are instant.

The UI is deliberately thin: every decision it displays is made by
submit/inference.py, the module the benchmark and the submission both use.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUBMIT = ROOT / "submit"
sys.path.insert(0, str(SUBMIT))

# Threads before onnxruntime is imported.
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
from insightface.utils import face_align  # noqa: E402

from inference import FacePipeline  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEMO_GALLERY = HERE / "demo_gallery"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pretty(pid: str) -> str:
    return pid.replace("_", " ").strip().title()


def to_bgr(rgb) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bbox_clip(f, img):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def face_metrics(f, img) -> dict:
    """Same quantities quality_ok() gates on, but returned for display."""
    kps = f.kps
    iod = float(np.linalg.norm(kps[0] - kps[1]))
    x1, y1, x2, y2 = bbox_clip(f, img)
    blur = 0.0
    if x2 > x1 and y2 > y1:
        gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    yaw = abs(float(kps[2][0] - (kps[0][0] + kps[1][0]) / 2)) / (iod + 1e-6) * 90
    return {"iod": iod, "blur": blur, "yaw": yaw, "px": (x2 - x1, y2 - y1)}


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, gallery_dir: Path, model: str = "buffalo_l"):
        self.gallery_dir = Path(gallery_dir)
        self.model_name = model
        self.cache_path = HERE / ".cache" / f"gallery_{model}.npz"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        DEMO_GALLERY.mkdir(exist_ok=True)

        print(f"loading {model} from {SUBMIT / 'models'} ...", flush=True)
        self.pipe = FacePipeline(model_name=model, models_root=str(SUBMIT),
                                 pad_models="default")

        self.ids: list[str] = []
        self.embs = np.zeros((0, 512), np.float32)
        self.sources: list[str] = []
        self.counts: list[int] = []
        self._load_or_build()

    # ---------- gallery ----------
    def _roots(self):
        return [r for r in (self.gallery_dir, DEMO_GALLERY) if r.exists()]

    def _files(self):
        out = []
        for root in self._roots():
            out += sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
        return sorted(out)

    def _signature(self) -> str:
        h = hashlib.sha1()
        h.update(self.model_name.encode())
        for p in self._files():
            # resolve() so a relative --gallery and the default absolute path
            # produce the same signature
            rp = p.resolve()
            st = rp.stat()
            h.update(f"{rp}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        return h.hexdigest()

    def _load_or_build(self):
        sig = self._signature()
        if self.cache_path.exists():
            z = np.load(self.cache_path, allow_pickle=False)
            if str(z["signature"]) == sig:
                self.ids = [str(x) for x in z["ids"]]
                self.embs = z["embs"].astype(np.float32)
                self.sources = [str(x) for x in z["sources"]]
                self.counts = [int(x) for x in z["counts"]]
                print(f"gallery: {len(self.ids)} identities loaded from cache", flush=True)
                return
            print("gallery cache is stale (images changed); rebuilding", flush=True)
        self._build(sig)

    def _build(self, sig: str | None = None):
        files = self._files()
        print(f"gallery: enrolling {len(files)} images (first run only) ...", flush=True)
        per: dict[str, list[tuple[Path, np.ndarray]]] = {}
        t0 = time.time()
        for i, p in enumerate(files, 1):
            emb, _reason = self.pipe.embed_image(p, quality_filter=True, gate_pad=False)
            if emb is not None:
                per.setdefault(p.parent.name, []).append((p, emb))
            if i % 100 == 0:
                print(f"  {i}/{len(files)}  ({(time.time()-t0)/i:.2f}s/img)", flush=True)

        self.ids, self.embs, self.sources, self.counts = [], [], [], []
        for pid in sorted(per):
            items = per[pid]
            m = np.mean(np.stack([e for _, e in items]), 0)
            m = m / (np.linalg.norm(m) + 1e-12)
            sims = np.stack([e for _, e in items]) @ m
            self.ids.append(pid)
            self.embs.append(m)
            self.sources.append(str(items[int(np.argmax(sims))][0]))
            self.counts.append(len(items))
        self.embs = (np.stack(self.embs).astype(np.float32) if self.embs
                     else np.zeros((0, 512), np.float32))
        self._save_cache(sig)
        print(f"gallery: {len(self.ids)} identities in {time.time()-t0:.0f}s", flush=True)

    def _save_cache(self, sig: str | None = None):
        np.savez(
            self.cache_path,
            ids=np.array(self.ids),
            embs=self.embs,
            sources=np.array(self.sources),
            counts=np.array(self.counts, dtype=np.int64),
            signature=np.array(sig or self._signature()),
        )

    def add_person(self, name: str, files: list[str] | None):
        if not name or not name.strip():
            return "Give the person a name first.", self.gallery_view(), self.summary()
        if not files:
            return "Attach at least one photo.", self.gallery_view(), self.summary()

        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "person"
        dest = DEMO_GALLERY / slug
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, src in enumerate(files):
            src = Path(src)
            if not src.exists():
                continue
            out = dest / f"{i:02d}{src.suffix.lower() or '.jpg'}"
            shutil.copy(src, out)
            saved.append(out)

        embs = []
        for p in saved:
            emb, _ = self.pipe.embed_image(p, quality_filter=True, gate_pad=False)
            if emb is not None:
                embs.append(emb)
        if not embs:
            return ("No usable face in those photos — nothing was enrolled. "
                    "Try a clearer, frontal photo."), self.gallery_view(), self.summary()

        m = np.mean(np.stack(embs), 0)
        m = m / (np.linalg.norm(m) + 1e-12)
        if slug in self.ids:
            k = self.ids.index(slug)
            merged = self.embs[k] + m
            self.embs[k] = merged / (np.linalg.norm(merged) + 1e-12)
            self.counts[k] += len(embs)
        else:
            self.ids.append(slug)
            self.embs = np.vstack([self.embs, m[None, :]]) if len(self.embs) else m[None, :]
            self.sources.append(str(saved[0]))
            self.counts.append(len(embs))
        self._save_cache()

        msg = (f"Enrolled **{pretty(slug)}** from {len(embs)} photo(s). "
               f"Gallery now holds {len(self.ids)} identities.")
        return msg, self.gallery_view(), self.summary()

    def gallery_view(self):
        return [(self.sources[i], f"{pretty(pid)} · {self.counts[i]} photos")
                for i, pid in enumerate(self.ids) if Path(self.sources[i]).exists()]

    def summary(self) -> str:
        return (f"**{len(self.ids)}** identities · "
                f"**{sum(self.counts)}** enrolment photos")

    # ---------- inference ----------
    def probe_faces(self, img: np.ndarray, gate: bool):
        """Mirror of FacePipeline.get_faces that keeps the intermediate state
        the UI needs (working canvas, per-face liveness, quality metrics)."""
        trace: list[str] = []
        work, faces = self.pipe._detect(img, trace=trace)
        out = []
        for f in faces:
            pad = self.pipe.spoof.predict(work, f.bbox) if self.pipe.spoof else None
            spoof = bool(pad and not pad["is_live"])
            ok, reason = self.pipe.quality_ok(f, work)
            emb = None if (gate and spoof) else self.pipe._embed(work, f)
            out.append({"face": f, "pad": pad, "spoof": spoof, "ok": ok,
                        "reason": reason, "emb": emb, "area": self.pipe._area(f),
                        "metrics": face_metrics(f, work)})
        return work, out, trace

    @staticmethod
    def best(items):
        good = [r for r in items if r["ok"] and r["emb"] is not None]
        salvage = [r for r in items if not r["ok"] and r["emb"] is not None]
        pool = good or salvage
        return max(pool, key=lambda r: r["area"]) if pool else None

    def topk(self, emb, k=5):
        if emb is None or len(self.ids) == 0:
            return []
        scores = self.embs @ emb
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i]), self.sources[i], self.counts[i])
                for i in order]

    def draw(self, img: np.ndarray, items, best, label: str) -> np.ndarray:
        out = img.copy()
        for r in items:
            x1, y1, x2, y2 = bbox_clip(r["face"], out)
            is_best = r is best
            if r["spoof"]:
                color = (60, 60, 220)
            elif is_best:
                color = (60, 180, 75)
            else:
                color = (150, 150, 150)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_best else 1)
            if is_best and label:
                text = label
                (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = max(y1 - 10, th + 8)
                cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 12, ty + base), color, -1)
                cv2.putText(out, text, (x1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
        return out


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
CSS = """
.verdict{padding:20px 24px;border-radius:16px;background:#0f172a;color:#e2e8f0;
  border-left:8px solid #16a34a;display:flex;align-items:center;gap:16px}
.verdict.unknown{border-left-color:#f59e0b}
.verdict.noface{border-left-color:#ef4444}
.verdict.spoof{border-left-color:#ef4444}
.v-kicker{font-size:12px;letter-spacing:.14em;color:#94a3b8;font-weight:700}
.v-name{font-size:32px;font-weight:700;line-height:1.15;margin-top:4px;color:#f8fafc}
.v-meta{font-size:14px;color:#94a3b8;margin-top:6px}
.v-score{margin-left:auto;font-size:44px;font-weight:700;color:#22c55e;
  font-variant-numeric:tabular-nums}
.verdict.unknown .v-score{color:#f59e0b}
.verdict.noface .v-score,.verdict.spoof .v-score{color:#ef4444}
.stat{font-size:13px;color:#475569}
"""


def verdict_html(kind: str, name: str, score: float, thr: float,
                 margin: float | None = None, note: str = "") -> str:
    kicker = {"match": "IDENTIFIED", "unknown": "UNKNOWN",
              "noface": "NO FACE", "spoof": "LIVENESS REJECTED"}[kind]
    meta = f"cosine {score:.3f} · threshold {thr:.2f}"
    if margin is not None and kind != "noface":
        meta += f" · margin {margin:+.3f} over #2"
    if note:
        meta += f" · {note}"
    return (f'<div class="verdict {kind}">'
            f'<div><div class="v-kicker">{kicker}</div>'
            f'<div class="v-name">{name}</div>'
            f'<div class="v-meta">{meta}</div></div>'
            f'<div class="v-score">{score:.2f}</div></div>')


def details_md(best, items, trace, top, thr, elapsed) -> str:
    if best is None:
        lines = ["#### Pipeline"]
        lines += [f"- {t}" for t in trace] or ["- no face detected"]
        if any(r["spoof"] for r in items):
            lines.append("- every detected face failed the liveness gate")
        return "\n".join(lines)

    f, m = best["face"], best["metrics"]
    pad = best["pad"]
    lines = [
        "#### This face",
        f"- **{m['px'][0]}×{m['px'][1]} px**, IOD **{m['iod']:.0f} px**, "
        f"blur **{m['blur']:.0f}**, yaw **~{m['yaw']:.0f}°**",
        f"- detector score **{f.det_score:.2f}**, quality **{best['reason']}**",
    ]
    if pad:
        gate = "rejected" if best["spoof"] else "passed"
        lines.append(f"- liveness **{pad['live_prob']:.3f}** "
                     f"(print {pad['p_print']:.3f}, replay {pad['p_replay']:.3f}) — {gate}")
    if best["emb"] is not None:
        lines.append(f"- embedding 512-d, ‖v‖ = {np.linalg.norm(best['emb']):.3f}")
    if top:
        lines.append(f"- best match **{top[0][0]}** at {top[0][1]:.3f}; "
                     f"threshold {thr:.2f}")
    lines += ["", "#### Pipeline"] + [f"- {t}" for t in trace]
    lines.append(f"- total {elapsed:.2f} s")
    return "\n".join(lines)


def top_rows(top, k=5):
    if not top:
        return []
    top1 = top[0][1]
    return [[i + 1, pretty(pid), f"{s:.4f}", f"{s - top1:+.4f}"]
            for i, (pid, s, _src, _n) in enumerate(top[:k])]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui(engine: Engine):
    n_ids, n_imgs = len(engine.ids), sum(engine.counts)

    def identify(rgb, gate, thr):
        if rgb is None:
            return (None, verdict_html("noface", "No image", 0.0, thr),
                    [], None, [], "Upload a photo to begin.")
        img = to_bgr(rgb)
        t0 = time.time()
        work, items, trace = engine.probe_faces(img, gate)
        best = engine.best(items)
        elapsed = time.time() - t0

        if best is None:
            kinds = [r for r in items if r["spoof"]]
            kind = "spoof" if kinds and not any(r["emb"] is not None for r in items) else "noface"
            html = verdict_html(kind, "Liveness rejected" if kind == "spoof" else "No face",
                                0.0, thr, note=f"{len(items)} face(s) in frame")
            return (to_rgb(engine.draw(work, items, None, "")), html, [], None, [],
                    details_md(None, items, trace, [], thr, elapsed))

        top = engine.topk(best["emb"], 5)
        pid, score, _src, _n = top[0]
        margin = score - top[1][1] if len(top) > 1 else None
        matched = score >= thr
        html = verdict_html("match" if matched else "unknown",
                            pretty(pid) if matched else "Unknown",
                            score, thr, margin,
                            note="quality rescued" if not best["ok"] else "")
        annotated = engine.draw(work, items, best, f"{pretty(pid)} {score:.2f}" if matched
                                else f"unknown {score:.2f}")
        crop = face_align.norm_crop(work, landmark=best["face"].kps, image_size=112)
        thumbs = [(src, f"{pretty(pid2)} · {s:.3f}")
                  for pid2, s, src, _ in top[:3] if Path(src).exists()]
        return (to_rgb(annotated), html, top_rows(top), to_rgb(crop),
                thumbs, details_md(best, items, trace, top, thr, elapsed))

    def verify(rgb_a, rgb_b, gate, thr):
        if rgb_a is None or rgb_b is None:
            return verdict_html("noface", "Two images needed", 0.0, thr), \
                   None, None, "Attach both photos."
        out = []
        for rgb in (rgb_a, rgb_b):
            work, items, trace = engine.probe_faces(to_bgr(rgb), gate)
            best = engine.best(items)
            out.append((best, work, items, trace))
        (ba, wa, ia, ta), (bb, wb, ib, tb) = out
        if ba is None or bb is None:
            which = "left" if ba is None else "right"
            return verdict_html("noface", f"No usable face on the {which}", 0.0, thr), \
                None, None, details_md(ba, ia, ta, [], thr, 0.0)
        score = float(np.dot(ba["emb"], bb["emb"]))
        same = score >= thr
        crop_a = face_align.norm_crop(wa, landmark=ba["face"].kps, image_size=112)
        crop_b = face_align.norm_crop(wb, landmark=bb["face"].kps, image_size=112)
        html = verdict_html("match" if same else "unknown",
                            "Same person" if same else "Different people",
                            score, thr)
        return (html, to_rgb(crop_a), to_rgb(crop_b),
                f"cosine **{score:.3f}** · threshold **{thr:.2f}** · "
                f"left quality **{ba['reason']}**, right quality **{bb['reason']}**")

    samples = []
    for pid in ("asim_munir", "imran_khan", "maryam_nawaz_sharif"):
        pdir = engine.gallery_dir.parent / "probe" / pid
        hits = sorted(pdir.glob("*")) if pdir.exists() else []
        if hits:
            samples.append([str(hits[0])])

    with gr.Blocks(title="TechFest · Face Recognition") as demo:
        gr.Markdown(
            "# Face recognition\n"
            f"**{n_ids}** identities · **{n_imgs}** enrolment photos · "
            "SCRFD + ArcFace · MiniFASNet liveness · CPU only · offline")

        with gr.Tabs():
            # ---------------- identify ----------------
            with gr.Tab("Identify"):
                with gr.Row():
                    with gr.Column(scale=5):
                        probe = gr.Image(type="numpy", sources=["upload", "webcam"],
                                         label="Probe photo", height=340)
                        gate = gr.Checkbox(
                            False,
                            label="Liveness gate — slower, rejects 2–3 % of genuine photos")
                        thr = gr.Slider(0.0, 0.8, value=0.40, step=0.01,
                                        label="Match threshold (cosine)")
                        go = gr.Button("Identify", variant="primary", size="lg")
                        gr.Examples(examples=samples, inputs=[probe],
                                    label="Try one of these")
                        gr.Markdown("<span class='stat'>Liveness score is shown whether or "
                                    "not the gate is on. Threshold 0.40 = FAR 0.48 % / "
                                    "FRR 0.69 %; set 0 for a closed-set benchmark.</span>")
                    with gr.Column(scale=7):
                        verdict = gr.HTML(verdict_html("noface", "Waiting for a photo", 0.0, 0.40))
                        annotated = gr.Image(label="Detected face", height=340,
                                             interactive=False)
                        table = gr.Dataframe(headers=["#", "identity", "cosine", "Δ top-1"],
                                             datatype=["number", "str", "number", "number"],
                                             interactive=False, wrap=True, label="Top 5")
                        with gr.Row():
                            crop = gr.Image(label="Aligned 112×112", width=150,
                                            height=150, interactive=False)
                            info = gr.Markdown("")
                        thumbs = gr.Gallery(label="Top matches in the gallery", columns=3,
                                            height=180, object_fit="cover")

                go.click(identify, [probe, gate, thr],
                         [annotated, verdict, table, crop, thumbs, info])
                probe.upload(identify, [probe, gate, thr],
                             [annotated, verdict, table, crop, thumbs, info])

            # ---------------- verify ----------------
            with gr.Tab("Verify 1:1"):
                gr.Markdown("Same pipeline, used as a verifier: two faces, one cosine.")
                with gr.Row():
                    img_a = gr.Image(type="numpy", sources=["upload", "webcam"],
                                     label="Photo A", height=280)
                    img_b = gr.Image(type="numpy", sources=["upload", "webcam"],
                                     label="Photo B", height=280)
                with gr.Row():
                    gate2 = gr.Checkbox(False, label="Liveness gate")
                    thr2 = gr.Slider(0.0, 0.8, value=0.40, step=0.01,
                                     label="Threshold (cosine)")
                    go2 = gr.Button("Compare", variant="primary")
                verdict2 = gr.HTML(verdict_html("noface", "Waiting for two photos", 0.0, 0.40))
                with gr.Row():
                    crop_a = gr.Image(label="A aligned", width=150, height=150,
                                      interactive=False)
                    crop_b = gr.Image(label="B aligned", width=150, height=150,
                                      interactive=False)
                info2 = gr.Markdown("")
                go2.click(verify, [img_a, img_b, gate2, thr2],
                          [verdict2, crop_a, crop_b, info2])

            # ---------------- gallery ----------------
            with gr.Tab("Gallery"):
                head = gr.Markdown(engine.summary())
                grid = gr.Gallery(value=engine.gallery_view(), columns=8, height=470,
                                  object_fit="cover", show_label=False)
                gr.Markdown("### Add someone\n"
                            "Enrol a new person on the spot — useful when a judge asks "
                            "“can it learn me?”. Photos are copied into `app/demo_gallery/` "
                            "and the gallery cache is updated immediately.")
                with gr.Row():
                    new_name = gr.Textbox(label="Name", placeholder="e.g. Ali Raza")
                    new_files = gr.File(label="Photos (1–5)", file_count="multiple",
                                        file_types=["image"])
                    add = gr.Button("Enrol", variant="primary")
                add_msg = gr.Markdown("")
                add.click(engine.add_person, [new_name, new_files],
                          [add_msg, grid, head])

            # ---------------- about ----------------
            with gr.Tab("How it works"):
                gr.Markdown(
                    "### Pipeline\n"
                    "1. **Detect** — SCRFD-10GF at 640 px. If nothing is found, retry "
                    "on a reflected-border canvas; if the only face is tiny, a second "
                    "320 px pass is merged in.\n"
                    "2. **Quality gate** — inter-ocular distance ≥ 64 px, blur ≥ 40, "
                    "|yaw| ≤ 30°. Failures are still embedded and used, but labelled.\n"
                    "3. **Liveness** — two MiniFASNet models (2.7× and 4.0× crop "
                    "margins) vote on print / live / replay. Optional gate; the score "
                    "is always shown.\n"
                    "4. **Align + embed** — 5-point `norm_crop` to 112×112, ArcFace "
                    "R50 → 512-d, L2-normalised.\n"
                    "5. **Match** — cosine against per-identity mean embeddings; "
                    "best above threshold wins.\n\n"
                    "### Numbers (REPORT.md)\n"
                    "| | |\n|---|---|\n"
                    "| Rank-1, 23 identities / 437 probes | **98.63 %** |\n"
                    "| EER | **0.69 %** |\n"
                    "| TAR @ FAR = 1e-3 | 97.71 % |\n"
                    "| Liveness APCER / BPCER (NUAA) | 1.33 % / 1.33 % |\n"
                    "| Runtime, CPU | ~0.3 s / image |\n\n"
                    "### Things worth trying in the demo\n"
                    "- A **profile or veiled** photo — watch the quality line and the "
                    "score drop.\n"
                    "- A **printed or screen photo** — tick the liveness gate and watch "
                    "it refuse to match.\n"
                    "- A **group photo** — the largest passing face is used.\n"
                    "- A **small face in a large frame** — the trace shows the 320 px "
                    "escalation firing.")

    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", default=str(ROOT / "data" / "pak_fig" / "gallery"),
                    help="gallery/<identity>/<image> root (default: the report gallery)")
    ap.add_argument("--model", default="buffalo_l", choices=["buffalo_l", "antelopev2"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true", help="public Gradio link")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    pack = SUBMIT / "models" / args.model
    if args.model != "buffalo_l" and not pack.exists():
        ap.error(f"{args.model} weights are not bundled ({pack}). "
                 f"The demo ships buffalo_l; see REPORT.md §12.4 for checksums.")

    engine = Engine(Path(args.gallery), model=args.model)
    demo = build_ui(engine)
    print(f"\n  {engine.summary().replace('**', '')}\n"
          f"  http://{args.host}:{args.port}\n", flush=True)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=not args.no_browser,
                show_error=True,
                # Gradio 6 moved these off the Blocks constructor
                theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
                css=CSS)


if __name__ == "__main__":
    main()
