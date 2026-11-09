# segrobust

Natural-corruption robustness evaluation for semantic segmentation models.

`segrobust` gives you three things that are awkward to get today:

1. **A maintained, deterministic implementation of the 19 ImageNet-C
   corruptions** (Hendrycks & Dietterich, ICLR 2019). The reference
   package (`imagecorruptions`) has not been released to PyPI since
   2020-12 and breaks with current scikit-image/scipy/Python 3.12
   (its issue #30). This library vendors the corruption code with the
   fixes applied, an optional (not required) numba dependency, and a
   `seed=` argument that makes every corruption a pure function of
   (image, name, severity, seed) without disturbing your global RNG.

2. **Small-image safety.** Every corruption runs cleanly down to 8x8
   images (upstream fails on CIFAR-sized 32x32 inputs; issues #13/#21),
   with no uninitialized-memory reads and no stdout noise.

3. **Segmentation-aware evaluation.** `corrupt_pair()` knows which
   corruptions are geometric vs photometric: photometric corruptions
   leave your label mask untouched; the one geometric corruption
   (elastic_transform) warps the mask with the *identical* displacement
   field (nearest-neighbor, so no invented labels). `evaluate()` runs
   your model over clean data and every corruption/severity pair and
   reports per-corruption mIoU, mean corrupt mIoU, and relative drop.

## Install

```bash
pip install "segrobust @ git+https://github.com/anishpentyala/segrobust.git"        # core
pip install "segrobust[fast] @ git+https://github.com/anishpentyala/segrobust.git"  # with numba acceleration for glass_blur
```

segrobust is not on PyPI yet; install from the repo as above (both forms
verified on a clean venv).

Requires Python 3.9+. Works with any framework: your model is just a
callable `uint8 image -> int label map`.

## Quick start

```python
import numpy as np
from segrobust import corrupt, corrupt_pair, evaluate, format_report

# one image, deterministically
img = ...                                # uint8 (H, W, 3)
foggy = corrupt(img, 'fog', severity=3, seed=0)

# image + mask, geometry handled for you
mask = ...                               # int labels (H, W)
c_img, c_mask = corrupt_pair(img, mask, 'elastic_transform',
                             severity=2, seed=1)

# full robustness report
def my_model(image):                     # your segmentation model
    ...                                  # -> int label map (H, W)

result = evaluate(my_model, images, masks, num_classes=21,
                  corruptions='common', severities=(1, 2, 3, 4, 5),
                  seed=0)
print(format_report(result))
```

Example report (illustrative output from a synthetic toy dataset with a
trivial intensity-rule model - NOT a benchmark result):

```
| corruption     | s1     | s3     | s5     | mean   |
|----------------|--------|--------|--------|--------|
| gaussian_noise | 0.8402 | 0.7082 | 0.5481 | 0.6988 |
...
clean mIoU: 1.0000  |  mean corrupt mIoU: 0.6703  |  relative drop: 0.3297  |  images: 4
```

## CLI

```bash
segrobust list                                        # corruptions + tags
segrobust corrupt in.png --name fog --severity 3 --seed 0 --out foggy.png
segrobust report result.json                          # markdown table
```

`format_report()` keeps severity labels exact: columns must be integers
from 1 through 5, and every corruption row must contain those exact settings.
The `report` command accepts the JSON produced by `json.dumps(evaluate(...))`:
integer column values and canonical string row keys (`"1"` through `"5"`).
It rejects fractional/boolean/string columns, duplicate JSON keys, alias keys
such as `"01"`, and missing or extra row settings instead of silently renaming
or dropping data. Valid reports preserve the requested unique column order. This
checks severity identity, not the accuracy or statistical validity of scores.

## Masked arrays require an explicit nodata choice

NumPy `MaskedArray` objects are rejected for images, target labels, model
predictions and confusion-count tables, including arrays whose mask is
currently all false. Ordinary NumPy conversion can discard the mask and score
underlying invalid pixels as real observations. Evaluation validates all input
images/targets before calling the model or progress callback, and rejects masked
predictions before converting them to arrays. No score is returned for an
incomplete evaluation.

Convert masked target labels deliberately to a plain integer array with an
unused, representable ignore label, then pass that same `ignore_index`:

```python
# Choose 255 only when it fits the target dtype and is not a real class.
target = masked_target.filled(255)
result = evaluate(model, images, [target], num_classes=2, ignore_index=255)
```

Do not discard a mask with `.data` unless including its underlying pixels is
what you intend. Images and predictions need their own explicit application
policy before evaluation; there is no automatic fill, masked prediction drop,
or masked-count interpretation. Plain arrays and existing `ignore_index`
behavior are unchanged. This guard does not make corruption or metrics
mask-aware, validate image nodata choices, or detect a mask removed by the caller.

## Metric label storage

Confusion-matrix bin arithmetic uses platform-width integer indices, not the
storage width of the masks. Valid uint8/int8 labels for 21 classes therefore
retain their exact class pairs instead of overflowing during multiplication.
Signed/unsigned integer arrays can be mixed. Ignored or out-of-range target
pixels are filtered before widening, including uint64 sentinel values.
Out-of-range predictions at valid target pixels still raise rather than being
clipped or dropped. Float, complex and object label arrays are rejected, not
silently truncated. Class counts must be positive integers with their square
inside the platform index range; this is an index-safety check, not a promise
that an arbitrarily large dense confusion matrix fits available memory.

## Design notes

- **Determinism.** With `seed` set, a corruption call seeds NumPy's
  legacy RNG for exactly the duration of the call and restores the
  previous state afterwards, so seeded calls reproduce byte-for-byte and
  never perturb the caller's random stream. `evaluate()` derives one
  stable draw per (image index, corruption, severity) triple, so reports
  reproduce across runs and processes.
- **Thread scope.** Calls through `corrupt` and `corrupt_pair` serialize their
  legacy NumPy RNG sections, including unseeded calls. Concurrent seeded
  library calls keep their own deterministic draws and restore the caller's
  saved RNG state. This does not isolate unrelated code that calls `np.random`
  directly in another thread, and is not a parallel-speedup claim. Separate
  processes are preferable when running expensive corruptions in parallel.
- **Photometric vs geometric.** 18 of the 19 corruptions are treated as
  photometric (mask unchanged) following the ImageNet-C label-preserving
  convention. `elastic_transform` genuinely moves pixels, so
  `corrupt_pair` warps the mask with the same field at order 0.
- **Provenance.** Corruption functions are adapted from
  [bethgelab/imagecorruptions](https://github.com/bethgelab/imagecorruptions)
  (Apache-2.0, see NOTICE) with fixes for current scikit-image/scipy,
  small inputs, and determinism. The frost textures are procedurally
  generated by this project (no third-party photo content, Apache-2.0);
  they replace the upstream photographic textures, whose per-asset
  licensing is undocumented, so frost's overlay pattern differs in
  appearance from the upstream implementation.
- **Benchmark comparability.** Corruptions implement the ImageNet-C
  definitions, but the fixes above (small-image guards, determinism,
  uninitialized-memory and dependency fixes, recompressed frost textures)
  mean outputs are NOT guaranteed bit-identical to the reference
  implementation. Compare numbers within your own study; do not quote
  them against published ImageNet-C/segmentation-C tables without
  re-running the reference on your data.

## Examples

`examples/deeplabv3_selfconsistency.py` runs segrobust against a real
pretrained model (torchvision DeepLabV3-ResNet50) on real COCO val2017
photos fetched at runtime. Labels are the model's own clean predictions
(pseudo-labels), so it measures prediction stability under corruption
(self-consistency), not ground-truth accuracy; `clean_miou` is 1.0 by
construction and the signal is the corrupt drop. Requires
`pip install torch torchvision`; about 6 minutes on CPU. Reference run
(4 photos, corruptions gaussian_noise/defocus_blur/fog/frost at
severities 1/3/5): mean corrupt mIoU 0.326; corrupt scores fall well below clean,
with severity-1 scores closest to clean (e.g. gaussian_noise 0.634 -> 0.060,
fog 0.240 -> 0.149; individual corruptions need not be monotone across
severities). Single model, four photos - a usability
demonstration, not a benchmark claim.

## Exact severity inputs

`corrupt`, `corrupt_pair` and `evaluate` accept integer-index severity values
from 1 through 5, including NumPy integer scalars. Floats, strings and booleans
are rejected rather than truncated or treated as an integer severity. Evaluation
validates its nonempty severity sequence before invoking the model or progress
callback. For example, `severities=[1.9]` raises instead of silently running
severity 1. This does not change the corruption formulas for valid integers.

## Exact mask labels under elastic warps

`corrupt_pair(..., "elastic_transform")` samples source-pixel addresses with
nearest-neighbor interpolation, then gathers labels from the original mask.
It does not convert label values to floating point. Large integer class IDs
and ignore labels therefore keep their exact bits, including signed and unsigned
64-bit labels. The displacement field and image interpolation are unchanged.
This is label-identity preservation, not a change to benchmark comparability.

## Tests

```bash
pip install -e ".[test]"
pytest            # 97 tests: every corruption x severities x small sizes,
                  # determinism, RNG hygiene, mask geometry, metrics, CLI
```

## License

Apache-2.0 (see LICENSE and NOTICE).

### Label-map validation

Masks passed to `corrupt_pair` and `evaluate` must be two-dimensional integer
(or boolean) NumPy arrays with exactly the image's height and width. Channel
axes such as `(H, W, 1)` and `(H, W, 3)` are rejected, not flattened into
additional scored pixels. Model predictions must have the exact mask shape.
`evaluate` validates all images and masks before calling the model or emitting
progress. Convert palette/RGB labels deliberately before evaluation; this
validation does not define a color-to-class mapping.

Evaluation plans are checked before model inference or progress callbacks:
`num_classes` must be a positive non-boolean integer within the platform index
range, and every explicitly listed corruption must be known. Invalid late plan
entries do not start a clean pass or run earlier corruptions first. These checks
do not estimate memory requirements or validate model behavior before execution.

### Confusion count tables

`iou_from_confusion` and `miou_from_confusion` require a square, two-dimensional
real numeric table of finite, nonnegative counts or weights. Invalid shapes,
negative counts and nonfinite values raise instead of producing a plausible
score. Fractional weights are supported. Integer counts are widened before
sums, and overflowing finite floating-point unions use per-class scaling.
Empty and absent classes keep their existing NaN behavior. This does not make
floating-point arithmetic exact or prevent an upstream accumulator from
already overflowing before it is passed to these functions.

## Per-class robustness diagnostics

Set `include_per_class=True` on `evaluate()` to inspect which classes lose IoU
under corruption. The optional `result["per_class"]` contains `clean` and
`per_corruption[name][severity]` records. Each record has `iou` and
`target_pixels` lists in class-index order (`0` through `num_classes - 1`):

```python
result = evaluate(my_model, images, masks, num_classes=21,
                  corruptions=["fog"], severities=[1, 3, 5],
                  include_per_class=True)
print(result["per_class"]["per_corruption"]["fog"][3]["iou"])
```

These diagnostics reuse each pass's accumulated confusion matrix, not an
average of per-image IoUs. They do not add model calls or change aggregate
scores. Ignored and out-of-range target pixels are excluded as before. A
class with neither target nor predicted pixels has NaN IoU. A target-absent
class with false-positive predictions has zero IoU and zero target pixels;
the existing mIoU still averages only target-present classes. This makes
the denominator visible without changing that metric's policy.

Defaults return the same result fields as before; `format_report()` remains
the aggregate table even with the optional diagnostics present. Lists use
plain Python numbers. Undefined scores remain float NaN, as in existing
aggregate results, so strict JSON consumers must handle those explicitly.
The option requires a Python bool and is checked before any model/progress
callback. Per-class lists increase report size with classes and settings;
this is not confidence intervals or evidence of benchmark generalization.

## Installed-package checks

The `Installed package` workflow builds a source distribution and then a wheel
from that distribution. It installs the wheel and runs the full test suite from
outside the checkout, checks the console entry point and installed version, and
checks that the requested optional backend is actually active. The Linux matrix
covers Python 3.9 without numba and Python 3.13 both without and with numba.
Dependencies are resolved for each runtime rather than pinned to a shared lock.
This is a regression gate for those environments, not proof of every dependency
combination, operating system or segmentation model.

## Unique experiment settings

`evaluate` requires unique corruption names and unique integer severity values.
Repeated entries previously executed extra model passes but overwrote earlier
scores in the keyed result. A repeated severity could also appear twice in a
report table, giving its row mean different weighting from the headline mean.
They now raise `ValueError` before any model or progress callback. Values are
not silently deduplicated: repeated trials need a separate explicit evaluation
plan, which this API does not model. `format_report` also rejects duplicate
columns or corruption rows in saved results. Unique input order is preserved,
and an empty corruption list still supports the existing clean-only evaluation.

## Model callback input ownership

Every model call receives a fresh writable copy of that condition's image.
In-place preprocessing inside the callable therefore cannot rewrite the input
dataset, affect later corruption draws, or change a target mask that happens
to share image storage. This also allows read-only NumPy dataset images. The
copy costs one image allocation per call; no extra inference is performed.
Returned prediction views remain supported and are scored immediately.

This is an input-ownership boundary, not a sandbox or model-state reset.
Callbacks must not mutate targets through captured references, retain and
concurrently rewrite returned prediction arrays, or change other caller state
if repeatable results are required. Model state and external concurrency are
not isolated.


Report row identity is checked as well as severity columns. `format_report`
and `segrobust report` require `per_corruption` to contain exactly the names
listed in `corruptions`; an unlisted saved row is not silently omitted.
A mismatch raises `ValueError` before any table is printed. Display order
still follows `corruptions`, not dictionary insertion order. Empty selections
remain valid when both the list and map are empty, including their existing
NaN summary. This checks row coverage, not the truth of saved scores or
whether the saved summary values were recomputed correctly.
