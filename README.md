<div align="center">

# StatCraft

**Second-Level Neuroimaging Analysis Tool**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Configuration](#configuration) | [Outputs](#outputs) | [License](#license)  | [Citation](#citation) | [Acknowledgments](#acknowledgments)

</div>



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

A pip-installable Python tool for second-level neuroimaging analysis, supporting group-level comparisons, method comparisons, and statistical inference on brain images (e.g., fMRI, PET). The tool adheres to BIDS standards, leverages Nilearn, and produces reproducible, interpretable results with minimal user input.

## Features

- **Multiple Analysis Types**
  - One-sample t-tests
  - Two-sample t-tests (group comparisons)
  - Paired t-tests (within-subject comparisons)
  - General Linear Model (GLM) with custom design matrices

- **BIDS-Compliant**
  - Reads data following BIDS structure
  - Supports filtering by BIDS entities (subject, session, task)
  - Works with derivatives from fMRIPrep, SPM, FSL, etc.

- **Statistical Inference**
  - Uncorrected thresholding (default p < 0.001)
  - FDR (False Discovery Rate) correction
  - FWER (Family-Wise Error Rate) via Bonferroni
  - Permutation-based FWER correction

- **Anatomical Annotation**
  - Harvard-Oxford atlas (default)
  - AAL, Destrieux, Schaefer atlases
  - Custom atlas support

- **Comprehensive Reporting**
  - HTML reports with visualizations
  - Design matrix plots
  - Activation maps (thresholded and unthresholded)
  - Cluster tables with anatomical labels

## Installation

```bash
git clone https://github.com/arovai/statCraft.git
cd statCraft
pip install -e .
```

## Quick Start

### Command Line Interface

#### Usage Patterns

StatCraft supports two input directory patterns:

**Pattern 1: BIDS Dataset with Separate Derivatives (Recommended)**
```bash
statcraft <BIDS_DIR> <OUTPUT_DIR> group --derivatives <DERIVATIVES_PATH> [options]
```
Use this when you have a clear separation between original BIDS data and processed derivatives (e.g., fMRIPrep output).

**Pattern 2: Derivatives-Only Directory**
```bash
statcraft <DERIVATIVES_PATH> <OUTPUT_DIR> group [--participants-file <PATH>] [options]
```
Use this when analyzing data directly from a derivatives folder without the original BIDS rawdata. You may need to specify the `--participants-file` path if it's not in the derivatives folder.

#### Examples

```bash
# One-sample t-test (Pattern 1: BIDS + Derivatives)
statcraft /data/bids /data/output group \
    --derivatives /data/derivatives/fmriprep \
    --analysis-type one-sample \
    --task nback

# One-sample t-test (Pattern 2: Derivatives-Only)
statcraft /data/derivatives/fmriprep /data/output group \
    --analysis-type one-sample \
    --task nback

# Derivatives-only with explicit participants file
statcraft /data/derivatives/fmriprep /data/output group \
    --participants-file /data/bids/participants.tsv \
    --analysis-type one-sample \
    --task nback

# Two-sample t-test (group comparison)
statcraft /data/bids /data/output group \
    --derivatives /data/derivatives/fmriprep \
    --analysis-type two-sample \
    --group-column group

# Paired t-test
statcraft /data/bids /data/output group \
    --derivatives /data/derivatives/fmriprep \
    --analysis-type paired \
    --pair-by session \
    --condition1 pre \
    --condition2 post

# Using a configuration file
statcraft /data/bids /data/output group \
    --derivatives /data/derivatives/fmriprep \
    --config config.yaml
```

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

# BIDS filters
bids_filters:
  task: nback
  session: baseline
  space: MNI152NLin2009cAsym

# Design matrix columns (from participants.tsv)
design_matrix:
  columns:
    - age
    - group
  add_intercept: true

# Contrasts to compute
contrasts:
  - age                    # Effect of age
  - patients - controls    # Group difference

# Paired test settings (for paired analysis)
paired_test:
  pair_by: session
  condition1: pre
  condition2: post

# Statistical inference
inference:
  alpha: 0.05
  height_threshold: 0.001
  cluster_threshold: 10
  corrections:
    - uncorrected
    - fdr
    - bonferroni

# Atlas for cluster annotation
atlas: harvard_oxford

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use StatCraft in your research, please cite:

```bibtex
@software{statcraft,
  title = {StatCraft: Second-Level Neuroimaging Analysis Tool},
  author = {StatCraft Contributors},
  year = {2024},
  url = {https://github.com/arovai/StatCraft}
}
```

## Acknowledgments

- [Nilearn](https://nilearn.github.io/) - Core neuroimaging functionality
- [PyBIDS](https://bids-standard.github.io/pybids/) - BIDS data handling
- [NiBabel](https://nipy.org/nibabel/) - NIfTI file handling
Neuroimaging Second Level analysis software
