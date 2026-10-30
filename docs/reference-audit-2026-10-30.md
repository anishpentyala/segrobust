# Reference audit 2026-10-30 (pre-0.1.1)

Audit of `src/segrobust/corruptions.py` against two sources:

- upstream `imagecorruptions` 1.1.2 sdist (PyPI, sha of extracted tree in
  `reference-audit-2026-10-30.diff` context)
- original Hendrycks & Dietterich `ImageNet-C/create_c/make_imagenet_c.py`
  (github.com/hendrycks/robustness, master pinned at
  8190fe329f5f072a06e3a2aea02bb5dda69aed9f, fetched 2026-10-30)

License verification (2026-10-30): both sources are Apache License 2.0
(hendrycks/robustness LICENSE fetched from the pinned revision;
imagecorruptions 1.1.2 declares Apache Software License). No verbatim
upstream source is republished here; provenance is recoverable from the
pinned URL above and the PyPI sdist. The diff embeds upstream context
lines under Apache-2.0 with attribution in the repo NOTICE. The in-test
reference copy in `tests/test_corruptions.py` is a derived excerpt of the
same Apache-2.0 sources, attributed in its docstring.

Method: unified diff of the vendored module against upstream 1.1.2
(`reference-audit-2026-10-30.diff`), then per-hunk classification against
the vendored module docstring's documented fix list, then three-way
verification of any undocumented hunk against the Hendrycks original.

## Inventory (23 hunks)

- 9 hunks: seed/rng threading (documented; `seed=` parameter plumbing)
- 5 hunks: skimage/scipy current-dependency compatibility (documented)
- 8 hunks: small-image guards (documented)
- 1 hunk: cosmetic reflow
- 1 hunk (gauss_function): honors the `mean` parameter upstream ignores;
  sole caller passes mean=0, so behavior is identical. Kept.
- 1 hunk (plasma_fractal filldiamonds): UNDOCUMENTED DIVERGENCE, fixed in
  0.1.1 (PR #4). See CHANGELOG.md. The vendored copy also used float32
  accumulation and uninitialized-memory-safe np.zeros; zeros kept
  (deterministic), float64 restored in 0.1.1.

## Bounded fidelity evidence (what is and is not established)

- plasma_fractal matches the reference algorithm EXACTLY at fixed seeds
  for mapsize in {8, 32, 64} (regression test, rtol/atol 1e-9). This
  validates the algorithm restoration at those sizes/seeds; it is not a
  proof of equivalence for all inputs, sizes, or RNG streams.
- fog end-to-end additionally involves the vendored severity tables and
  skimage/scipy pipeline, which carry documented compatibility changes.
- frost intentionally differs in appearance: textures are procedurally
  generated (see NOTICE), not the upstream photographic assets.
- small-input guards change behavior deliberately below reference
  minimums. Outputs are NOT guaranteed bit-identical to the reference
  implementation (see README "Benchmark comparability").
- Internal test-pass alone never established generator equivalence; the
  plasma_fractal divergence is the evidence.
