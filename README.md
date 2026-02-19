<div align="center">

# StatCraft

**Second-Level Neuroimaging Analysis Tool**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Configuration](#configuration) | [Outputs](#outputs) | [License](#license)  | [Citation](#citation) | [Acknowledgments](#acknowledgments)

</div>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

CLI tool for second-level neuroimaging analysis, supporting group-level comparisons, method comparisons, and statistical inference on brain images (e.g., fMRI, PET) or connectivity matrices. Uses flexible pattern matching for file discovery, leverages Nilearn, and produces reproducible, interpretable results with minimal user input.

## Features

- **Multiple Analysis Types**
  - One-sample t-tests
  - Two-sample t-tests (group comparisons)
  - Paired t-tests (within-subject comparisons)
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
python -m venv
source venv/bin/activate
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
statcraft <INPUT_DIR> <OUTPUT_DIR> --analysis-type one-sample --pattern "PATTERN"
```
The pipeline will search for files matching `PATTERN` in the `<INPUT_DIR>` (exploring subfolders), perform the one-sample t-test, and save the results in `<OUTPUT_DIR>`.

*Example:*

A one-sample t-test on maps from first-level `beta1` maps with `task-motor` in the name:
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output --analysis-type one-sample --pattern "*task-motor*beta1*.nii.gz"
```

**Smoothing Brain Map Data**

When analyzing brain maps (e.g., statistical maps from first-level analyses), spatial smoothing can improve signal-to-noise ratio and increase statistical power. Use the `--smoothing` option to specify the smoothing strength in mm (FWHM - Full Width at Half Maximum):

```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> --analysis-type one-sample --pattern "PATTERN" --smoothing 6
```

The `--smoothing` parameter:
- Accepts values in mm (e.g., 4, 6, 8)
- Default is 0 (no smoothing)
- Only applies to brain map analysis (NIfTI images), not connectivity matrices
- Uses spatial Gaussian smoothing via nilearn's `SecondLevelModel`

*Example with smoothing:*

```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output --analysis-type one-sample --pattern "*task-motor*beta1*.nii.gz" --smoothing 6
```

**Two-sample t-tests**

The logic is similar as above, except that now one must specify two patterns to define the two groups to compare. We can also assign these groups custom names to ease the output filenaming:
```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> --analysis-type two-sample --patterns "Group1=PATTERN1 Group2=PATTERN2"
```
The strings `Group1` and `Group2` are arbitrary and are used in the output filenaming and in the analysis report.

*Example:*

If you have distinguishable names between the two groups you want to compare (e.g. `c001, c002, ...` for "controls" and `p001, p002, ...` for "patients"):
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output --analysis-type two-sample --patterns "controls=sub-c*.nii.gz patients=sub-p*.nii.gz"
```
Of course, you can also use the GLM approach to have groups defined in a spreadsheet (see below), which has the advantage of being independent of your participant-naming choices.

**Paired t-tests**

This case is similar to the two-sample case except that one must provide a key to pair the data across the two groups. We do this by using the `--pair-by` argument:
```bash
statcraft <INPUT_DIR> <OUTPUT_DIR> --analysis-type paired --patterns "Group1=PATTERN1 Group2=PATTERN2" --pair-by "ENTITY"
```
The pairing is done using BIDS-like `key-value` structures in filenames. For instance:
- `--pair-by "sub"` (default): Files with `sub-001`, `sub-002`, etc. will be paired together
- `--pair-by "ses"`: Files with `ses-pre`, `ses-post`, etc. will be paired together
- `--pair-by "run"`: Files with `run-1`, `run-2`, etc. will be paired together

Supports both BIDS abbreviations (`sub`, `ses`, `run`, `task`, etc.) and full names (`subject`, `session`, `run`, etc.).

*Note:* `--pair-by` is optional, with default value `sub`.

*Examples:*

Compare maps from session 1 to session 2, paired by subjects (using default `--pair-by sub`):
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output --analysis-type paired --patterns "session1=*ses-1*.nii.gz session2=*ses-2*.nii.gz"
```

Pair by session instead of subject (if you have multiple sessions and want to pair conditions within sessions):
```bash
statcraft /path/to/first_level_fmri_analyzes /path/to/output --analysis-type paired --patterns "pre=*ses-pre*.nii.gz post=*ses-post*.nii.gz" --pair-by "ses"
```

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
statcraft <INPUT_DIR> <OUTPUT_DIR> --analysis-type glm --participant-files <PATH_TO_PARTICIPANTS.TSV> --regressors sex age IQ --categorical-regressors sex --contrasts age M-F --pattern "*stat-effect*.nii.gz
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

- [Nilearn](https://nilearn.github.io/) - Core neuroimaging functionality
- [PyBIDS](https://bids-standard.github.io/pybids/) - BIDS data handling
- [NiBabel](https://nipy.org/nibabel/) - NIfTI file handling
