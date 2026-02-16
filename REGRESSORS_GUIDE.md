# Regressors & Contrasts Guide

This guide covers how to specify regressors (covariates) and contrasts in StatCraft for both NIfTI and connectivity analyses.

## Overview

When running a `glm` analysis type, you can include columns from `participants.tsv` as regressors in the design matrix. StatCraft handles:

- **Continuous regressors** (e.g., age, IQ): z-scored by default (can be disabled)
- **Categorical regressors** (e.g., sex, group): dummy-coded into indicator columns

---

## CLI Options

### `--regressors`

Specify one or more column names from `participants.tsv` to include in the design matrix.

```bash
--regressors age --regressors sex --regressors IQ
```

### `--categorical-regressors`

Explicitly declare which regressors are categorical (dummy-coded). If not specified, StatCraft auto-detects based on data type and number of unique values (≤5).

```bash
--categorical-regressors sex --categorical-regressors group
```

### `--no-standardize-regressors`

Disable z-scoring of continuous regressors. By default, continuous regressors are standardized to mean=0, std=1. This flag keeps them in their original units.

```bash
--no-standardize-regressors
```

> **Note:** This is independent of `--zscore`, which z-scores brain maps at the subject level.

### `--contrast`

Specify the contrast expression to test. Contrast expressions reference design matrix column names.

```bash
--contrast 'age'           # Effect of age
--contrast 'sex_M-sex_F'   # Males vs Females
```

---

## Design Matrix Column Naming

Understanding how columns are named in the design matrix is key to writing contrasts:

| Regressor type | Input column | Design matrix columns | Notes |
|---|---|---|---|
| Continuous | `age` | `age` | z-scored by default |
| Categorical | `sex` (values: M, F) | `sex_M`, `sex_F` | One column per level (`drop_first=False`) |
| Categorical | `group` (values: HC, PAT) | `group_HC`, `group_PAT` | One column per level |
| Intercept | (automatic) | `intercept` | Added by default |

So with `--regressors age --regressors sex --categorical-regressors sex`, the design matrix columns would be:

```
intercept | age | sex_F | sex_M
```

---

## Contrast Specification

Contrasts are mathematical expressions that reference design matrix column names. They use [nilearn's expression parser](https://nilearn.github.io/stable/modules/generated/nilearn.glm.expression_to_contrast_vector.html).

### Examples

| Goal | Contrast expression | Explanation |
|---|---|---|
| Effect of age | `age` | Tests whether age has a significant effect |
| Males vs Females | `sex_M-sex_F` | Difference between male and female groups |
| Mean across groups | `0.5*sex_M+0.5*sex_F` | Average effect across both sexes |
| Intercept (mean) | `intercept` | Tests whether the mean effect differs from 0 |

### Default contrast

If no `--contrast` is specified, StatCraft defaults to testing the intercept (`mean_effect` = `[1, 0, 0, ...]`).

---

## Full CLI Examples

### 1. Effect of age on connectivity (NIfTI)

```bash
statcraft /data/bids /data/output \
  -d /data/derivatives/fmriprep \
  -t glm \
  --regressors age \
  --contrast age \
  -p '*task-rest*bold*.nii.gz'
```

### 2. Sex differences in connectivity matrices

```bash
statcraft /data/bids /data/output \
  -d /data/derivatives \
  --data-type connectivity \
  -t glm \
  --regressors sex \
  --categorical-regressors sex \
  --contrast 'sex_M-sex_F' \
  -p '**/*_connmat.npy'
```

### 3. Age effect controlling for sex

```bash
statcraft /data/bids /data/output \
  -d /data/derivatives \
  --data-type connectivity \
  -t glm \
  --regressors age --regressors sex \
  --categorical-regressors sex \
  --contrast age \
  -p '**/*_connmat.npy'
```

### 4. Multiple regressors without standardization

```bash
statcraft /data/bids /data/output \
  -d /data/derivatives/fmriprep \
  -t glm \
  --regressors age --regressors IQ \
  --no-standardize-regressors \
  --contrast age \
  -p '*task-rest*bold*.nii.gz'
```

### 5. Simple one-sample test (no regressors)

```bash
statcraft /data/bids /data/output \
  -d /data/derivatives \
  --data-type connectivity \
  -t one-sample \
  -p '**/*_connmat.npy'
```

---

## YAML Configuration

All CLI options can also be set in a config file:

```yaml
analysis_type: glm

design_matrix:
  columns:
    - age
    - sex
  categorical_columns:
    - sex
  standardize_continuous: true   # set to false to disable z-scoring
  add_intercept: true

# Contrast expression or list of contrasts
contrast: "age"

# Or multiple contrasts:
# contrasts:
#   - "age"
#   - expression: "sex_M-sex_F"
#     name: "sexDifference"
```

---

## Output Naming

Output files include the contrast name using BIDS-compatible naming:

```
output_dir/
├── data/
│   ├── {bids_prefix}_contrast-effectOfAge_stat-tstat.npy
│   ├── {bids_prefix}_contrast-effectOfAge_stat-pval.npy
│   ├── {bids_prefix}_contrast-effectOfAge_stat-tstat.json
│   ├── {bids_prefix}_contrast-effectOfAge_stat-fdr_threshold.npy
│   └── {bids_prefix}_contrast-effectOfAge_stat-bonferroni_threshold.npy
├── tables/
│   ├── {bids_prefix}_contrast-effectOfAge_stat-fdr_edges.tsv
│   └── {bids_prefix}_contrast-effectOfAge_stat-bonferroni_edges.tsv
└── {bids_prefix}_contrast-effectOfAge_report.html
```

The contrast name is auto-generated from the expression:
- `age` → `effectOfAge`
- `sex_M-sex_F` → `sex_MVersusSex_f`
- Or provide a custom name via the YAML config's `name` field
