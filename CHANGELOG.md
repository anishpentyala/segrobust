# Changelog

## Unreleased

- Reject report condition lists that omit or invent saved `per_corruption`
  rows before rendering. Valid subsets, display order, empty selections and
  NaN scores keep their existing behavior. This checks row coverage, not
  score truth or saved summary arithmetic.


### Changed

- Reject NumPy MaskedArray images, labels, model predictions and confusion
  tables instead of silently discarding invalid-pixel masks. All-false masks
  are also unsupported. This intentionally changes previously accepted input
  behavior. Convert nodata explicitly; for target labels, use an unused
  representable integer fill label with the same ignore_index. AI-assisted
  implementation, tests and documentation.

### Release plan

- This source change is unreleased. The package and runtime version remain
  0.1.1, matching the latest recorded release section below. Before a future
  package release, choose and update both versions together and include this
  input-compatibility change in its release notes. This change does not create
  a tag, publish a package, or schedule or authorize a release.

- Reject channel-axis and noninteger masks before corruption or evaluation; require exact two-dimensional prediction shape rather than silently scoring extra channels. AI-assisted implementation and tests.

## 0.1.1 (2026-10-30)

### Fixed

- **fog / plasma_fractal now matches the ImageNet-C reference algorithm.**
  A reference audit of the vendored corruptions against upstream
  imagecorruptions 1.1.2 and the original Hendrycks & Dietterich
  implementation found that the initial vendoring of `plasma_fractal`
  (v0.1.0) diverged from both references: the left-diamond step dropped
  the `ulgrid +` term from both accumulated sums and wrote its means onto
  the corner grid instead of the left-diamond positions. Consequently
  `fog` outputs in v0.1.0 differed from the ImageNet-C fog definition.
  The diamond-square step is restored to the reference formulation
  (keeping the small-input `mapsize >= 8` guard, deterministic
  zero-initialization, and float64 accumulation), and a regression test
  now pins `plasma_fractal` against a verbatim copy of the reference
  algorithm at fixed seeds. Fog outputs change versus v0.1.0; rerun
  studies that used v0.1.0 fog. The other 18 corruptions were audited
  against the same references with no further divergences found.

## Confusion-matrix index width

- Widen validated, filtered integer label operands before bin multiplication
  and addition. Avoid wrong scores from narrow uint8/int8/uint16/int16 masks
  and float promotion from mixed signed/unsigned label storage.
- Reject noninteger label arrays and invalid or index-overflowing class counts
  without truncating labels. Preserve ignore filtering and prediction errors.
