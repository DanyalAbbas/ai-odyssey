# TechFest Face Recognition — submission package

Self-contained, offline, CPU-only. The hidden benchmark is run by the organisers;
this package is the thing they run.

## Run it

```bash
pip install -r requirements.txt

# 1:N identification
python predict.py --gallery <gallery_dir> --probe <probe_dir> --output predictions.csv

# 1:1 verification
python predict.py --pairs <pairs.csv> --output predictions.csv
```

Nothing is downloaded at runtime. Model weights are in `models/`.

## Expected input

**Gallery** — one subdirectory per identity, images inside:

```
gallery/
  ahsan_iqbal/        *.jpg
  bilawal_bhutto_zardari/  *.jpg
```

Flat gallery instead? Pass an index:

```
python predict.py --gallery imgs/ --gallery-index index.csv --probe probe/
# index.csv:  file,id   (header optional; columns may be in either order)
```

**Probe** — a directory (searched recursively), or a single image.
**Pairs** — CSV whose first two columns are image paths. A header row is
tolerated; extra columns are ignored.

## Output

Identification:

```
probe,predicted_id,score,quality,match
subdir/img.jpg,ahsan_iqbal,0.8665,ok,True
subdir/bad.jpg,unknown,0.0,no_face,False
```

Verification:

```
pair,left,right,score,match
1,a.jpg,b.jpg,0.7123,True
```

`quality` records why a face was accepted or rejected: `ok`, `ok_padded`
(recovered by the tight-crop retry), `fallback_quality`, `spoof_<p>`,
`low_iod_<px>`, `blur_<var>`, `yaw_<deg>`, `no_face`. Rows are never dropped, so
the output always has exactly one line per input.

## Design notes

**Detection has two escalations**, both of which were needed on the public
benchmark:
1. nothing detected → retry on a 30 % reflected-border copy (tight headshots give
   SCRFD no context);
2. only tiny faces detected → second pass at 320 px and NMS merge (a small
   background face can otherwise win over the actual subject in a portrait).

**Liveness is off by default.** The MiniFASNet gate rejects ~2–4 % of genuine
faces, which on an attack-free recognition benchmark can only lower the score.
Enable it with `--pad` when presentation attacks are in scope; it costs about
0.5 s/image on CPU.

**Threshold** is 0.40 cosine. Being conservative costs ~2 correct probes in 437.

## Models

| `--model` | Files | Size | Notes |
|---|---|---|---|
| `buffalo_l` (default) | `models/buffalo_l/{det_10g,w600k_r50}.onnx` | 184 MB | R50, ~0.3 s/image CPU |
| `onnx` | `models/adaface_ir101.onnx` | 249 MB | higher score margin, ~1 s/image |
| `antelopev2` | not bundled | +260 MB | R100, ~3× slower than R50 |

`antelopev2` is supported in code but its weights are not shipped, to keep the
package under 500 MB. Drop them in `models/antelopev2/` to enable it.

If the benchmark is known to be closed-set (every probe has a mate), pass
`--threshold 0`: rejection can then only lose points. At the shipped 0.40, two
correct probes in 437 are answered `unknown`.

```bash
python predict.py --gallery g/ --probe p/ --model onnx \
  --onnx models/adaface_ir101.onnx --output predictions.csv
```

## Runtime

CPU, single process, 8 threads: roughly **0.3 s/image** for detection +
R50 embedding, i.e. ~5 min per 1,000 images. Memory stays under 1.5 GB. R100/
IR-101 is ~3× slower. `--threads N` sets the thread count.
