# Reproducible paired comparisons of participant maps

StatCraft compares maps already produced by participant-level tools such as
CVRMap. It does not repeat BOLD denoising or physiological modeling. Keep the
participant maps and their processing sidecars: a significant difference of
method means does not by itself establish agreement, accuracy, or superiority.

## Select one complete observation per participant

Use restrictive patterns that select one processing variant, spatial template,
resolution, session and run for each named method. The default pairing key is
`sub`. Both ordinary paired differences and `--combine` reject duplicate matches
before any arithmetic. Missing pairing entities are errors. Missing method maps
exclude the whole participant, with a warning.

For matching methods in the same session/run, `--pair-by sub,ses,run` requires
every selected map to have the same values for all three entities. YAML also
accepts `pair_by: [subject, session, run]`. Omit entities that intentionally differ
between conditions: compare session pre and post with `--pair-by sub`, selecting
each session in its own pattern.

A group t-test assumes independent participants. Composite keys do not turn
repeated runs or sessions into independent subjects: StatCraft rejects repeated
subjects among the retained observations. Select one run/session or aggregate
within participant upstream. Repeated-measures models beyond one paired
observation per participant are outside this workflow.

## Choose an inference domain independently of normalization

Three masks serve different purposes:

| Setting | Purpose |
|---|---|
| `--inference-mask PATH_OR_GLOB` | Anatomical domain in which group hypotheses are tested |
| `--scaling KEY=PATH_OR_GLOB` | ROI used to calculate the mean divisor for each participant map |
| `--zscore --mask PATH_OR_GLOB` | ROI used to calculate each map's mean and standard deviation |

Choose the inference domain before examining the method differences. A common
brain mask or a prespecified ROI is recommended. A fixed path uses one mask for
all observations; a quoted glob matches per-observation masks using exact BIDS
entity values. Ambiguous matches fail. All masks must already match the input
maps' three-dimensional shape and affine (absolute tolerance `1e-5`, no relative
tolerance). Resample intentionally upstream and record that operation; StatCraft
does not silently resample these masks.

The effective inference mask is the intersection of the anatomical masks and
finite (`not NaN/Inf`) support of **every original included map**. It is computed
before normalization and before differences/combinations. A zero map value or a
zero difference remains valid. Scaling and z-score ROIs never select the testing
domain. Without `--inference-mask`, subject-specific `*desc-brain_mask.nii`
or `.nii.gz` files next to each input are matched by BIDS entities. If any
mask is missing, StatCraft warns and uses Nilearn's MNI152 brain mask instead
of the individual masks, resampled to the input grid using nearest-neighbor
interpolation and intersected with finite input support. This fallback assumes
MNI152 registration, which grid checks cannot verify, and does not guarantee
individual coverage. Inspect the saved mask and JSON provenance; for a manuscript
analysis, prefer a prespecified mask reflecting the actual participants.

The exact same effective mask is used by the GLM, uncorrected/FDR/Bonferroni
thresholding and permutation inference. Values outside that mask are zeroed
before subtraction and optional smoothing, so nonfinite voxels cannot spread into
the test domain. An empty intersection fails. Constant zero differences remain
in the domain; Nilearn may warn that they contribute no information to the
permutation null distribution.

## Geometry and normalization validation

All selected maps must be 3D and have a common shape and affine. Geometry checks
are repeated before arithmetic. An incompatible map excludes the complete pair
(or complete set in combine mode), including otherwise valid partner maps,
before normalization. The excluded-file log records both the offending map and
its partner(s). Fewer than two remaining independent pairs is an error. Geometry
agreement does not establish anatomical registration quality: inspect alignment
separately.

Before normalization, StatCraft intersects every included original map's finite
support and every matched normalization ROI. The resulting shared normalization
support is identical across methods and participants and remains independent of
the inference mask. Scaling calculates each map's mean on that shared support;
empty/zero-mean ROIs fail. Z-scoring calculates each map's mean and standard
deviation on the same shared support and fails for empty/zero-standard-deviation
ROIs. Normalization masks must be
finite and nonnegative. Nonfinite values produced within the inference domain
fail rather than being silently replaced. Means/standard deviations are
calculated separately for each map, so normalization changes the scientific
quantity being compared; specify its rationale in the manuscript.

### Use the inference mask for whole-brain normalization

Supply the same mask path (or the same per-observation mask glob) to
`--inference-mask` and `--scaling brain=...` when both operations should use the
same domain. For paired group analysis, the effective domain is the intersection
across **all included maps and participants**, not a different mask for each
participant pair. If several contrasts should share exactly the same domain,
construct one intersection over the complete comparison set and supply that
fixed mask to every contrast. Retain the input masks and report coverage loss.
This domain is the common covered brain, which may be smaller than a full
anatomical whole-brain mask. Valid zero and negative estimates are not excluded.

Mean scaling can become unstable for signed maps whose positive and negative
values cancel. Configure an optional safeguard in the YAML file:

```yaml
normalization:
  mean_guard:
    minimum_absolute_mean_ratio: 0.01
    require_positive_mean: true
```

The signed diagnostic is `q = mean(values) / mean(abs(values))`, calculated on
the effective normalization mask. With these settings, normalization requires
`q >= 0.01`; zero and nonfinite means also fail. This is invariant to positive
changes of units. The example bounds the mean absolute normalized magnitude at
100; it does **not** bound every voxel, establish adequate SNR or define a
physiological exclusion threshold. A failing map stops the analysis with an
error, without dropping a pair, clipping values or substituting another scale.
Inspect the map and mask before deciding what to do.

The general-purpose defaults are ratio `0.0` and positive-mean requirement
`false`, preserving exact-zero-only rejection and support for negative-mean maps.
Thus the stronger CVR policy must be explicitly selected. The paired inference
mask JSON includes `MeanNormalizationQC` entries for each input map: signed mean,
mean absolute magnitude, signed ratio, effective voxel count and guard settings.
The same quantities are logged. Setting the identical mask for inference and
scaling does not make normalization implicit; specify both settings.

## Command-line examples

The positional analysis level `group` is required. Input patterns are relative to
the input/derivative roots; mask paths/globs may be absolute. Keep output folders
outside the searched input roots to avoid rediscovering previous results.

Compare two relative CVR variants from the same participant/session/run:

```bash
statcraft /data/cvrmap /results/gs_vs_veins group \
  --analysis-type paired \
  --patterns 'GS=**/*ses-01*run-1*proc-gsVariant*desc-cvr_bold.nii.gz Veins=**/*ses-01*run-1*proc-veinVariant*desc-cvr_bold.nii.gz' \
  --pair-by sub,ses,run \
  --inference-mask /data/masks/common_brain_mask.nii.gz \
  --scaling brain=/data/masks/common_brain_mask.nii.gz \
  --contrasts GS-Veins \
  --permutation --n-permutations 5000 --n-jobs 4
```

The `proc-gsVariant`/`proc-veinVariant` tokens are placeholders: replace them
with the exact `proc` values from the CVRMap outputs for the denoising/probe/lag
recipes you intend to compare. Select those processing identities explicitly;
a broad `*desc-cvr_bold.nii.gz` can match several variants and will fail as
ambiguous. Current CVRMap manifests distinguish the cohort-wide `ProcessingVariant`
recipe from participant-specific `ResultIdentitySHA256` input provenance. Confirm
that the chosen recipe ID is shared across the intended cohort and inspect its
`ProcessingRecipe`; earlier development outputs used participant-specific `proc`
IDs and must be regenerated or selected in separate, audited recipe directories.
StatCraft does not infer scientific compatibility from the filename token alone.
The participant-map suffix is `desc-cvr_bold.nii.gz`. If the gas and resting scans
have different task/run labels, pair by `sub` (or `sub,ses`) and choose each task
and run in its corresponding pattern. Put both maps on a scientifically justified
common relative scale before comparing; absolute `%BOLD/mmHg` and uncalibrated
resting-state surrogate coefficients are not directly interchangeable.

Compare a third variant with the average of two alternatives:

```bash
statcraft /data/cvrmap /results/combined_methods group \
  --analysis-type paired \
  --patterns 'GS=**/*proc-gsVariant*desc-cvr_bold.nii.gz Veins=**/*proc-veinVariant*desc-cvr_bold.nii.gz Arteries=**/*proc-arteryVariant*desc-cvr_bold.nii.gz' \
  --pair-by sub \
  --combine '0.5*GS + 0.5*Veins - Arteries' \
  --inference-mask '/data/masks/sub-*_space-MNI152NLin2009cAsym_brain_mask.nii.gz'
```

Every declared pattern must appear in the expression. A missing input excludes
the whole observation. Define this contrast before inspecting results; it is a
test of a mean linear combination, not an agreement statistic.

Equivalent configuration for an ordinary paired comparison:

```yaml
analysis_type: paired
inference_mask: /data/masks/common_brain_mask.nii.gz
paired_test:
  pair_by: [subject, session, run]
  sample_patterns:
    GS: '**/*ses-01*run-1*proc-gsVariant*desc-cvr_bold.nii.gz'
    Veins: '**/*ses-01*run-1*proc-veinVariant*desc-cvr_bold.nii.gz'
scaling_key: brain
scaling_pattern: /data/masks/common_brain_mask.nii.gz
normalization:
  mean_guard:
    minimum_absolute_mean_ratio: 0.01
    require_positive_mean: true
inference:
  alpha_corrected: 0.05
  alpha_uncorrected: 0.001
  two_sided: true
  run_permutation: true
  n_permutations: 5000
random_state: 42
```

```bash
statcraft /data/cvrmap /results/gs_vs_veins group --config comparison.yaml \
  --permutation --n-permutations 5000
```

Explicit CLI options override config values; omitted options preserve the
configuration, including analysis type, pairing keys, masks, permutation count,
thresholds, smoothing and report settings. `--no-permutation` and `--no-zscore`
can disable those configured operations. `random_state` is available in the
configuration. The resolved saved configuration records the values actually
used. Use `--atlas none --no-report` for a numeric-only run without atlas
downloads or HTML generation.

## Inspect and retain the outputs

Alongside the group maps, StatCraft always saves `*_desc-inference_mask.nii.gz`
and its JSON sidecar, independently of `--save-supplementary-data`. With
normalization enabled, it also saves the distinct `*_desc-normalization_mask.nii.gz`.
The JSON sidecar records both policies, voxel counts, original input paths,
explicit inference masks and normalization mask sources.
Retain these files with the resolved config, design matrix, discovered/valid/
excluded input lists and input CVRMap sidecars. A new processing variant should
use a separate output directory. Re-run previous paired inference when its
input-selection or mask behavior may have been affected by earlier versions.

## Tests for developers

```bash
python -m pip install -e '.[dev]'
python -m pytest tests
```

The paired-route tests use tiny synthetic images and exercise public dispatch,
actual file discovery, pair exclusion, arithmetic, Nilearn GLM and correction
families, permutation inference, mask export and CLI parsing. Numerical results
are checked against independent array arithmetic and SciPy t-tests. Atlas
downloads and HTML generation are excluded from these numerical tests.
