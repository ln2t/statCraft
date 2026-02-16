# StatCraft - Second-Level Neuroimaging Analysis Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

### From PyPI (recommended)

```bash
pip install statcraft
```

### From Source

```bash
git clone https://github.com/ln2t/StatCraft.git
cd StatCraft
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/ln2t/StatCraft.git
cd StatCraft
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# One-sample t-test
statcraft /path/to/bids /path/to/output \
    --derivatives /path/to/derivatives \
    --analysis-type one-sample \
    --task nback

# Two-sample t-test (group comparison)
statcraft /path/to/bids /path/to/output \
    --derivatives /path/to/derivatives \
    --analysis-type two-sample \
    --group-column group

# Paired t-test
statcraft /path/to/bids /path/to/output \
    --derivatives /path/to/derivatives \
    --analysis-type paired \
    --pair-by session \
    --condition1 pre \
    --condition2 post

# Using a configuration file
statcraft /path/to/bids /path/to/output \
    --derivatives /path/to/derivatives \
    --config config.yaml

# Skip BIDS validation
statcraft /path/to/bids /path/to/output \
    --derivatives /path/to/derivatives \
    --analysis-type one-sample \
    --skip-bids-validator
```

### Generate Default Configuration

```bash
statcraft --init-config config.yaml
```

### Python API

```python
from statcraft import StatCraftPipeline

# Create and run pipeline
pipeline = StatCraftPipeline(
    bids_dir="/path/to/bids",
    output_dir="/path/to/output",
    derivatives=["/path/to/derivatives"],
    config={
        "analysis_type": "one-sample",
        "bids_filters": {"task": "nback"},
        "contrasts": ["intercept"],
    }
)

results = pipeline.run()
```

## Configuration

StatCraft can be configured via command-line arguments, configuration files (YAML/JSON), or Python dictionaries.

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

## Data Requirements

### BIDS Structure

Your data should follow the BIDS standard:

```
dataset/
├── participants.tsv          # Required: participant metadata
├── participants.json
├── rawdata/                   # BIDS rawdata (can be same as dataset root)
│   ├── sub-01/
│   │   └── ...
│   └── sub-02/
│       └── ...
└── derivatives/
    └── fmriprep/              # Or any other pipeline output
        ├── sub-01/
        │   └── func/
        │       └── sub-01_task-nback_space-MNI152NLin2009cAsym_stat-effect_statmap.nii.gz
        └── sub-02/
            └── ...
```

### participants.tsv

The `participants.tsv` file must contain a `participant_id` column and any variables used in the analysis:

```tsv
participant_id	age	group	sex
sub-01	25	patient	M
sub-02	30	control	F
sub-03	28	patient	M
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

## API Reference

### Core Classes

- `StatCraftPipeline`: High-level pipeline for running complete analyses
- `DataLoader`: BIDS-compliant data loading and validation
- `DesignMatrixBuilder`: Design matrix construction
- `SecondLevelGLM`: GLM fitting and contrast computation
- `StatisticalInference`: Thresholding and cluster analysis
- `ClusterAnnotator`: Anatomical labeling with atlases
- `ReportGenerator`: HTML report generation

### Example: Custom Analysis

```python
from statcraft.core.data_loader import DataLoader
from statcraft.core.design_matrix import DesignMatrixBuilder
from statcraft.core.glm import SecondLevelGLM
from statcraft.core.inference import StatisticalInference
from statcraft.core.annotation import ClusterAnnotator

# Load data
loader = DataLoader(
    bids_dir="/path/to/bids",
    derivatives=["/path/to/derivatives"],
    output_dir="/path/to/output",
)
images = loader.get_images(bids_filters={"task": "nback"})
valid_images, _ = loader.validate_mni_space(images)

# Build design matrix
participants = loader.get_participants_for_images(valid_images)
dm_builder = DesignMatrixBuilder(participants)
design_matrix = dm_builder.build_design_matrix(columns=["age", "group"])
dm_builder.add_contrast("age")

# Fit model
image_paths = [str(img["path"]) for img in valid_images]
glm = SecondLevelGLM()
glm.fit(image_paths, design_matrix)
glm.compute_contrast(dm_builder.contrasts["effectOfAge"], contrast_name="effectOfAge")

# Run inference
inference = StatisticalInference(alpha=0.05, height_threshold=0.001)
inference.threshold_fdr(glm.get_stat_map("effectOfAge"), contrast_name="effectOfAge")

# Annotate clusters
annotator = ClusterAnnotator(atlas="harvard_oxford")
annotated_table = annotator.annotate_cluster_table(
    inference.get_cluster_table("effectOfAge", "fdr")
)
```

## CLI Commands

```bash
# Run analysis
statcraft run BIDS_DIR OUTPUT_DIR --derivatives DERIV_DIR [options]

# Initialize configuration
statcraft init-config OUTPUT_PATH [--format yaml|json]

# Validate data
statcraft validate BIDS_DIR [--derivatives DERIV_DIR]

# Show version and dependencies
statcraft info
```

## Dependencies

- Python >= 3.8
- nilearn >= 0.10.0
- nibabel >= 4.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pybids >= 0.15.0
- jinja2 >= 3.0.0
- pyyaml >= 6.0
- click >= 8.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
Neuroimaging Second Level analysis software
