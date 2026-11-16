<div align="center">

# StatCraft

**Second-Level Neuroimaging Analysis Tool**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Configuration](#configuration) | [Outputs](#outputs) | [License](#license)  | [Citation](#citation) | [Acknowledgments](#acknowledgments)

</div>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

StatCraft is a BIDS-friendly wrapper around [Nilearn](https://nilearn.github.io/) for second-level neuroimaging analysis. It supports group-level comparisons, method comparisons, and statistical inference on brain images (e.g., fMRI, PET) or connectivity matrices. Under the hood, the core statistical routines — GLM fitting, contrast computation, permutation testing and thresholding — rely on Nilearn's well-proven implementations. StatCraft adds flexible pattern-based file discovery and produces reproducible, interpretable results with minimal user input.

## Features

- **Multiple Analysis Types**
  - One-sample t-tests
  - Two-sample t-tests (group comparisons)
  - Paired t-tests (within-subject comparisons), including a generalized mode combining any number of maps with a custom linear combination
  - General Linear Model (GLM) with custom design matrices

- **Flexible File Discovery**
  - Pattern-based file matching with glob patterns
  - Participant filtering by subject ID
  - Works with derivatives from any first-level analysis
  - Supports BIDS-like and custom file naming

- **Statistical Inference**
  - Uncorrected thresholding
  - FDR (False Discovery Rate) correction
  - FWER (Family-Wise Error Rate) via Bonferroni
  - Permutation-based FWER correction

- **Anatomical Annotation of Clusters**
  - Harvard-Oxford atlas (default)
  - AAL, Destrieux, Schaefer atlases
  - Custom atlas support

- **Comprehensive Reporting**
  - HTML reports with visualizations
  - Design matrix plots
  - Activation maps (thresholded and unthresholded)
  - Cluster tables with anatomical labels

## Installation

We strongly recommend to use a virtual environment to work with this project

```bash
git clone https://github.com/ln2t/statCraft.git
cd statCraft
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Command Line Interface

#### Usage Patterns

StatCraft uses flexible pattern matching for file discovery. The idea is that you point to one or more directories and define filenaming patterns using wildcards, and this defines the data on which the second-level analysis will be performed.

This allows the user to use this tool in a variety of first-level tools without any particular constraints on the output structure.

Here are the typical usage, depending on the analysis type:

**One-sample t-tests**

```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> group --analysis-type one-sample --pattern "PATTERN"
```
The pipeline will search for files matching `PATTERN` in the `<INPUT_DIR>` (exploring subfolders), perform the one-sample t-test, and save the results in `<OUTPUT_DIR>`.

*Example:*

A one-sample t-test on maps from first-level `beta1` maps with `task-motor` in the name:
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type one-sample --pattern "*task-motor*beta1*.nii.gz"
```

**Smoothing Brain Map Data**

When analyzing brain maps (e.g., statistical maps from first-level analyses), spatial smoothing can improve signal-to-noise ratio and increase statistical power. Use the `--smoothing` option to specify the smoothing strength in mm (FWHM - Full Width at Half Maximum):

```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> group --analysis-type one-sample --pattern "PATTERN" --smoothing 6
```

The `--smoothing` parameter:
- Accepts values in mm (e.g., 4, 6, 8)
- Default is 0 (no smoothing)
- Only applies to brain map analysis (NIfTI images), not connectivity matrices
- Uses spatial Gaussian smoothing via nilearn's `SecondLevelModel`

*Example with smoothing:*

```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type one-sample --pattern "*task-motor*beta1*.nii.gz" --smoothing 6
```

**Two-sample t-tests**

The logic is similar as above, except that now one must specify two patterns to define the two groups to compare. We can also assign these groups custom names to ease the output filenaming:
```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> group --analysis-type two-sample --patterns "Group1=PATTERN1 Group2=PATTERN2"
```
The strings `Group1` and `Group2` are arbitrary and are used in the output filenaming and in the analysis report.

*Example:*

If you have distinguishable names between the two groups you want to compare (e.g. `c001, c002, ...` for "controls" and `p001, p002, ...` for "patients"):
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type two-sample --patterns "controls=sub-c*.nii.gz patients=sub-p*.nii.gz"
```
Of course, you can also use the GLM approach to have groups defined in a spreadsheet (see below), which has the advantage of being independent of your participant-naming choices.

**Paired t-tests**

This case is similar to the two-sample case except that one must provide a key to pair the data across the two groups. We do this by using the `--pair-by` argument:
```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> group --analysis-type paired --patterns "Group1=PATTERN1 Group2=PATTERN2" --pair-by "ENTITY"
```
The pairing is done using BIDS-like `key-value` structures in filenames. For instance:
- `--pair-by "sub"` (default): Files with `sub-001`, `sub-002`, etc. will be paired together
- `--pair-by "sub,ses,run"`: Match all three entity values across methods in a selected session/run.
- Entities that differ between conditions (for example pre/post session) must be selected in the patterns, not included in the matching key.

Supports both BIDS abbreviations (`sub`, `ses`, `run`, `task`, etc.) and full names (`subject`, `session`, `run`, etc.).

*Note:* `--pair-by` is optional, with default value `sub`. Every pattern must match exactly one file per complete key; duplicates and missing keys are errors. Repeated observations from the same subject are not independent and must be selected/aggregated upstream.

All image-analysis routes use and save a brain inference mask, independently of
`--scaling`/`--mask` normalization ROIs. By default StatCraft looks for
subject-specific `*desc-brain_mask.nii` or `.nii.gz` files in each input map's
directory, matching BIDS entities. Use `--inference-mask PATH_OR_GLOB` to select
a fixed mask or individual masks elsewhere. It intersects every included mask
and the finite support of the original input maps; valid zero values remain.
Masks must match the maps' grid. Ambiguous matches and empty intersections fail.
If any individual mask is missing, a warning is emitted and the Nilearn MNI152
brain mask is resampled by nearest neighbor to the input grid and used instead
of all individual masks. This assumes the maps are registered to MNI152;
matching shape and affine alone do not establish registration. Check the saved
inference mask and its JSON provenance, particularly when using the fallback.
Connectivity analyses are unaffected. See [paired method comparisons](docs/paired_method_comparisons.md)
for geometry/exclusion behavior, common normalization support, CVRMap examples,
reproducible configurations and limitations.

*Examples:*

Compare maps from session 1 to session 2, paired by subjects (using default `--pair-by sub`):
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type paired --patterns "session1=*ses-1*.nii.gz session2=*ses-2*.nii.gz"
```

For pre/post sessions, match subjects while selecting each session explicitly:
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type paired --patterns "pre=*ses-pre*.nii.gz post=*ses-post*.nii.gz" --pair-by sub \
    --inference-mask /path/to/common_brain_mask.nii.gz
```

**Custom linear combination of maps (generalized paired test)**

The paired test can be generalized to combine any number of per-participant maps using a custom linear combination, instead of always computing a simple two-sample difference. Define as many named patterns as needed with `--patterns`, then describe how to combine them with `--combine`:
```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> group --analysis-type paired --patterns "myMap1=PATTERN1 myMap2=PATTERN2 myMap3=PATTERN3" --combine "myMap1 + 0.5*myMap2 - myMap3"
```
The `--combine` expression references the names defined in `--patterns` and supports `+`, `-`, and `coefficient*name` terms (e.g., `0.5*myMap2`). For each participant, the corresponding maps are combined voxel-wise according to this formula, and the resulting per-participant map is brought to a group-level one-sample t-test (exactly like the classic paired test on difference images).

As with the classic paired test, pairing across maps is done using BIDS-like `key-value` entities in filenames (`--pair-by`, default `sub`). Each pattern must resolve to **exactly one** map per participant:
- If a pattern matches **more than one** file for a given participant, this is an error.
- If a pattern matches **no** file for a given participant, a warning is issued and that participant is excluded from the analysis.

*Example with 3 maps:*
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type paired \
    --patterns "myMap1=*task-a*stat-effect*.nii.gz myMap2=*task-b*stat-effect*.nii.gz myMap3=*task-c*stat-effect*.nii.gz" \
    --combine "myMap1 + 0.5*myMap2 - myMap3"
```

*Example with 4 maps:*
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --analysis-type paired \
    --patterns "myMap1=*task-a*stat-effect*.nii.gz myMap2=*task-b*stat-effect*.nii.gz myMap3=*task-c*stat-effect*.nii.gz myMap4=*task-d*stat-effect*.nii.gz" \
    --combine "myMap1 + myMap2 - myMap3 - myMap4"
```

*This feature is also fully configurable from a config file*, without needing any of `--patterns`/`--combine`/`--pair-by` on the command line. Simply set `paired_test.sample_patterns` and `paired_test.combine` in your YAML/JSON config (see the [example config with 3 maps](examples/config_paired_combine.yaml)):
```yaml
analysis_type: paired

paired_test:
  pair_by: sub
  sample_patterns:
    myMap1: "*task-a*stat-effect*.nii.gz"
    myMap2: "*task-b*stat-effect*.nii.gz"
    myMap3: "*task-c*stat-effect*.nii.gz"
  combine: "myMap1 + 0.5*myMap2 - myMap3"
```
then run:
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output group --config config.yaml
```
Command-line options (`--patterns`, `--combine`, `--pair-by`, `--exclude`, ...) always take precedence over values loaded from the config file.

**General Linear Model (GLM)**

For GLM analysis one must provide a design matrix. This is done by passing the `--participants-file`, which follows the structure of the `participants.tsv` file in a BIDS directory:

```
participant_id  sex age (other columns)
sub-001 F 42  ...
sub-CTL1 F 93  ...
sub-abcd M 17  ...
```
By default, a design matrix is built using all the columns of this file. If only a subset of columns should be included, this can be achieved using the `--regressors` option. Moreover, columns that should be treated as categorical can be specified with `--categorical-regressors` (dummy-coded in the analysis). Finally, the contrast to compute is defined using the `--contrasts` argument.

Here is a example featuring this functionality:
```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> group --analysis-type glm --participants-file <PATH_TO_PARTICIPANTS.TSV> --regressors sex age IQ --categorical-regressors sex --contrasts age M-F --pattern "*stat-effect*.nii.gz"
```
In this example: columns `sex`, `age` and `IQ` are used to build the design matrix, with `sex` being treated as a categorical variable, and two contrasts are computed: the effect of `age` as well as the difference between `M` and `F` (assuming those are the labels in original `participants.tsv` file).

*Notes*: 
- By default, non-categorical variables are z-scored. To control this, one can use `--no-standardize-regressors`, e.g. `--no-standardize-regressors IQ`
- An intercept is also added in the design matrix. Contrasts using the intercept can be built using the word `mean`, e.g. `--contrasts mean`. To remove the intercept, use `--no-intercept` option.
- Non-trivial contrasts can be built using simple operands, e.g. `--contrasts 0.5*M+0.5*F-mean`

## Configuration

StatCraft can be configured via command-line arguments, configuration files (YAML/JSON), or Python dictionaries.

### Generate Default Configuration

```bash
statcraft --init-config config.yaml
```

### Example Configuration (YAML)

```yaml
# Analysis type
analysis_type: glm

# Participant filtering (optional)
participant_label: null  # Or: ["01", "02", "03"]

# File pattern with embedded filters
# Include task, session, space directly in the pattern
file_pattern: "**/*task-nback*ses-baseline*space-MNI152*stat-effect*.nii.gz"

# Design matrix columns (from participants.tsv)
design_matrix:
  columns:
    - age
    - group
  add_intercept: true
  categorical_columns:
    - group

# Contrasts to compute
contrasts:
  - age                    # Effect of age
  - patients - controls    # Group difference

# Paired test settings (for paired analysis)
paired_test:
  pair_by: sub
  sample_patterns:
    pre: "**/*ses-pre*.nii.gz"
    post: "**/*ses-post*.nii.gz"

# Statistical inference
inference:
  alpha_corrected: 0.05       # Significance for corrected thresholds
  alpha_uncorrected: 0.001    # Cluster-forming threshold
  cluster_threshold: 10
  corrections:
    - uncorrected
    - fdr
    - bonferroni

# Atlas for cluster annotation
atlas: harvard_oxford

# Smoothing for brain map analysis (in mm FWHM)
# Use 0 for no smoothing (default)
# Only applies to brain map analysis, not connectivity matrices
smoothing_fwhm: 0

# Output settings
output:
  generate_report: true
  report_filename: report.html
```

## Outputs

Output names use the task entity from the included input filenames. When
multiple tasks are present, their distinct names are sorted and concatenated,
followed by `H` and the first 12 hexadecimal characters of the SHA-256 hash of
the sorted task names joined with a NUL byte (for example,
`task-task1task2H<hash>`). The hash distinguishes ambiguous concatenations;
it is a short identifier, not a guarantee against hash collisions. If some input
filenames lack a task entity, StatCraft warns that the label represents only the
known tasks. A single task retains its original name.

StatCraft generates:

1. **Statistical Maps** (`*.nii.gz`)
   - T-statistic maps
   - P-value maps
   - Effect size maps
   - Thresholded maps (uncorrected, FDR, Bonferroni)

2. **Cluster Tables** (`*.tsv`)
   - Peak coordinates (MNI)
   - Cluster sizes
   - Peak statistics
   - Anatomical labels

3. **HTML Report** (`report.html`)
   - Methodology description
   - Design matrix visualization
   - Activation maps
   - Glass brain views
   - Cluster tables

4. **Configuration** (`config.yaml`)
   - Complete configuration for reproducibility

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Citation

If you use StatCraft in your research, please cite:

```bibtex
@software{statcraft,
  title = {StatCraft: Second-Level Neuroimaging Analysis Tool},
  author = {StatCraft Contributors},
  year = {2024},
  url = {https://github.com/ln2t/StatCraft}
}
```

## Acknowledgments

StatCraft is built on top of [Nilearn](https://nilearn.github.io/), which provides the core statistical and neuroimaging routines (second-level GLM, contrast computation, thresholding, and atlas-based annotation). We are grateful to the Nilearn developers and contributors for maintaining such a robust and well-documented library.

Other key dependencies:

- [PyBIDS](https://bids-standard.github.io/pybids/) - BIDS data handling
- [NiBabel](https://nipy.org/nibabel/) - NIfTI file handling
