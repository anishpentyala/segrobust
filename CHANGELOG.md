# Changelog

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
