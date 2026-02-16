"""
Configuration handling for StatCraft.

This module handles:
- Configuration file parsing (YAML/JSON)
- Configuration validation
- Default values
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# Default configuration values
DEFAULT_CONFIG = {
    # BIDS filters
    "bids_filters": {
        "task": None,
        "session": None,
        "subject": None,
        "space": "MNI152NLin2009cAsym",
    },

    # File pattern for finding images
    "file_pattern": None,

    # Exclude pattern for filtering out unwanted files
    "exclude_pattern": None,

    # Sample patterns for multi-sample analyses (two-sample, paired)
    "sample_patterns": None,

    # Design matrix specification
    "design_matrix": {
        "columns": [],
        "add_intercept": True,
        "categorical_columns": None,
        "standardize_continuous": True,
    },

    # Analysis type
    "analysis_type": "glm",  # "one-sample", "two-sample", "paired", "glm"

    # Contrasts
    "contrasts": [],

    # Group comparison settings (for two-sample)
    "group_comparison": {
        "group_column": None,
        "group1": None,
        "group2": None,
    },

    # Paired test settings
    "paired_test": {
        "pair_by": None,  # e.g., "session"
        "condition1": None,  # e.g., "pre"
        "condition2": None,  # e.g., "post"
        "sample_patterns": None,  # e.g., {"GS": "*GS*cvr*.nii.gz", "SSS": "*SSS*cvr*.nii.gz"}
    },

    # Scaling settings
    "scaling_key": None,  # Key for output filenames (e.g., "brain")
    "scaling_pattern": None,  # Mask pattern or path for scaling

    # Z-scoring settings
    "zscore": False,  # Whether to apply z-scoring
    "mask_pattern": None,  # Brain mask pattern for z-scoring (e.g., '/path/to/sub-*/*/*brain*mask.nii.gz')

    # Statistical inference settings
    # NOTE: Two different significance levels (alpha) are used:
    # - alpha_uncorrected (default: 0.001): Significance level for uncorrected analysis.
    #   Also called "cluster-forming threshold" - defines which voxels are considered
    #   active for initial cluster detection.
    # - alpha_corrected (default: 0.05): Significance level for multiple comparison
    #   corrected analyses (FDR, Bonferroni, permutation). Controls the family-wise
    #   or false discovery error rate.
    "inference": {
        "alpha_corrected": 0.05,  # Significance level for corrected thresholds (FDR, Bonferroni, permutation)
        "alpha_uncorrected": 0.001,  # Significance level for uncorrected analysis (cluster-forming threshold)
        "cluster_threshold": 10,
        "two_sided": True,
        "corrections": ["uncorrected", "fdr", "bonferroni"],
        "run_permutation": False,
        "n_permutations": 1000,
    },

    # GLM settings
    "glm": {
        "smoothing_fwhm": 5.0,  # Smoothing kernel FWHM in mm (set to None to disable)
    },

    # Atlas for cluster annotation
    "atlas": "harvard_oxford",

    # Output settings
    "output": {
        "save_maps": True,
        "save_tables": True,
        "generate_report": True,
        "report_filename": "report.html",
    },

    # Computational settings
    "n_jobs": 1,
    "random_state": None,
    "verbose": 1,
}


class Config:
    """
    Configuration manager for StatCraft.
    
    Parameters
    ----------
    config_file : str or Path, optional
        Path to configuration file (YAML or JSON).
    **kwargs
        Additional configuration options to override defaults.
    
    Attributes
    ----------
    data : dict
        Configuration dictionary.
    """
    
    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        # Start with defaults
        self.data = self._deep_copy(DEFAULT_CONFIG)
        
        # Load from file if provided
        if config_file is not None:
            self.load_from_file(config_file)
        
        # Override with kwargs
        self._update_nested(self.data, kwargs)
        
        # Validate configuration
        self.validate()
    
    def _deep_copy(self, d: Dict) -> Dict:
        """Create a deep copy of a dictionary."""
        import copy
        return copy.deepcopy(d)
    
    def _update_nested(self, base: Dict, updates: Dict) -> None:
        """Update nested dictionary with another dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """
        Load configuration from a YAML or JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to configuration file.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        logger.info(f"Loading configuration from: {filepath}")
        
        with open(filepath, "r") as f:
            if filepath.suffix in [".yaml", ".yml"]:
                file_config = yaml.safe_load(f)
            elif filepath.suffix == ".json":
                file_config = json.load(f)
            else:
                # Try YAML first, then JSON
                try:
                    file_config = yaml.safe_load(f)
                except Exception:
                    f.seek(0)
                    file_config = json.load(f)
        
        if file_config is not None:
            self._update_nested(self.data, file_config)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to a YAML or JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to output file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            if filepath.suffix in [".yaml", ".yml"]:
                yaml.safe_dump(self.data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(self.data, f, indent=2)
        
        logger.info(f"Configuration saved to: {filepath}")
    
    def validate(self) -> None:
        """
        Validate the configuration.
        
        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        errors = []
        
        # Validate analysis type
        valid_types = ["glm", "one-sample", "two-sample", "paired", "one_sample", "two_sample"]
        if self.data["analysis_type"] not in valid_types:
            errors.append(f"Invalid analysis_type: {self.data['analysis_type']}. Must be one of {valid_types}")
        
        # Validate paired test settings
        if self.data["analysis_type"] == "paired":
            paired = self.data["paired_test"]
            if not paired.get("pair_by"):
                errors.append("paired_test.pair_by is required for paired analysis")

            # Check if using old method (condition1/condition2) or new method (sample_patterns)
            has_sample_patterns = paired.get("sample_patterns") is not None
            has_conditions = paired.get("condition1") and paired.get("condition2")

            if not has_sample_patterns and not has_conditions:
                errors.append("paired_test requires either (condition1 and condition2) or sample_patterns")

        # Validate two-sample settings
        if self.data["analysis_type"] == "two-sample":
            # Check if using patterns or group_column
            has_sample_patterns = self.data.get("sample_patterns") is not None
            has_group_column = self.data["group_comparison"].get("group_column") is not None

            if not has_sample_patterns and not has_group_column:
                errors.append("two-sample analysis requires either sample_patterns or group_comparison.group_column")
        
        # Validate inference settings
        inference = self.data["inference"]
        if not 0 < inference["alpha_corrected"] < 1:
            errors.append(f"alpha_corrected must be between 0 and 1, got {inference['alpha_corrected']}")
        if not 0 < inference["alpha_uncorrected"] < 1:
            errors.append(f"alpha_uncorrected must be between 0 and 1, got {inference['alpha_uncorrected']}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports dot notation).
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., "inference.alpha").
        default : any
            Default value if key not found.
        
        Returns
        -------
        any
            Configuration value.
        """
        keys = key.split(".")
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key (supports dot notation).
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., "inference.alpha").
        value : any
            Value to set.
        """
        keys = key.split(".")
        data = self.data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data
    
    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self._deep_copy(self.data)
    
    def summary(self) -> str:
        """
        Get a text summary of the configuration.
        
        Returns
        -------
        str
            Configuration summary.
        """
        lines = ["Configuration Summary", "=" * 40]
        
        lines.append(f"\nAnalysis Type: {self.data['analysis_type']}")
        
        lines.append("\nBIDS Filters:")
        for k, v in self.data["bids_filters"].items():
            if v is not None:
                lines.append(f"  {k}: {v}")
        
        if self.data["contrasts"]:
            lines.append(f"\nContrasts: {len(self.data['contrasts'])}")
            for c in self.data["contrasts"]:
                lines.append(f"  - {c}")
        
        lines.append(f"\nInference Settings:")
        lines.append(f"  Alpha (corrected): {self.data['inference']['alpha_corrected']}")
        lines.append(f"  Alpha (uncorrected): {self.data['inference']['alpha_uncorrected']}")
        lines.append(f"  Corrections: {', '.join(self.data['inference']['corrections'])}")
        
        lines.append(f"\nAtlas: {self.data['atlas']}")
        
        return "\n".join(lines)


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Config:
    """
    Load configuration from file and/or keyword arguments.
    
    Parameters
    ----------
    config_file : str or Path, optional
        Path to configuration file.
    **kwargs
        Additional configuration options.
    
    Returns
    -------
    Config
        Configuration object.
    """
    return Config(config_file=config_file, **kwargs)


def create_default_config(output_path: Union[str, Path]) -> Path:
    """
    Create a comprehensive, well-documented default configuration file.

    Parameters
    ----------
    output_path : str or Path
        Path for the output configuration file.

    Returns
    -------
    Path
        Path to created configuration file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Comprehensive template with extensive documentation
    template = """# ===============================================================================
# StatCraft Configuration File
# ===============================================================================
# This file contains all possible configuration options for StatCraft, a second-
# level neuroimaging analysis tool for group-level statistical inference.
#
# For each option, we provide:
#   - Description of what it does
#   - CLI equivalent (if available)
#   - Example values
#   - Valid options/range
#
# USAGE:
#   statcraft --config this_file.yaml
#   OR
#   statcraft /path/to/bids /path/to/output -d /path/to/derivatives --config this_file.yaml
#
# For more information, visit: https://github.com/arovai/StatCraft
# ===============================================================================

# -------------------------------------------------------------------------------
# PATHS AND DIRECTORIES
# -------------------------------------------------------------------------------
# These can be specified in the config file OR as command-line arguments.
# CLI arguments take precedence over config file values.

# Path to BIDS rawdata directory
# CLI equivalent: BIDS_DIR (positional argument)
# Example: bids_dir: /data/rawdata/my-study
bids_dir: null

# Path for analysis outputs
# CLI equivalent: OUTPUT_DIR (positional argument)
# Example: output_dir: /data/derivatives/statcraft
output_dir: null

# Path(s) to derivative folder(s)
# CLI equivalent: --derivatives / -d (can be specified multiple times)
# Examples:
#   derivatives: ["/data/derivatives/fmriprep"]
#   derivatives:
#     - "/data/derivatives/fmriprep"
#     - "/data/derivatives/cvrmap"
derivatives: []

# -------------------------------------------------------------------------------
# ANALYSIS TYPE
# -------------------------------------------------------------------------------
# Specifies the type of second-level analysis to perform.
# CLI equivalent: --analysis-type / -t
#
# Valid options:
#   - "one-sample": One-sample t-test (mean activation vs zero)
#   - "two-sample": Independent two-sample t-test (group comparison)
#   - "paired": Paired t-test (within-subject comparison)
#   - "glm": General Linear Model with custom design matrix
#
# Examples:
#   analysis_type: one-sample
#   analysis_type: two-sample
#   analysis_type: paired
#   analysis_type: glm
analysis_type: glm

# -------------------------------------------------------------------------------
# BIDS FILTERS
# -------------------------------------------------------------------------------
# Filter input images based on BIDS entities. These filters are applied when
# discovering images in the derivatives folder using PyBIDS.
#
# NOTE: When using file_pattern or sample_patterns, BIDS filters are applied
# AFTER pattern matching. It's often better to include filters directly in
# your glob patterns.
bids_filters:
  # Task label to filter images
  # CLI equivalent: --task
  # Example: task: nback
  # Set to null to include all tasks
  task: null

  # Session label to filter images
  # CLI equivalent: --session
  # Example: session: 01
  # Set to null to include all sessions
  session: null

  # Subject ID(s) to include in analysis
  # CLI equivalent: --subject (can be specified multiple times)
  # Examples:
  #   subject: null              # Include all subjects
  #   subject: "01"              # Include only sub-01
  #   subject: ["01", "02"]      # Include sub-01 and sub-02
  subject: null

  # Spatial normalization space
  # CLI equivalent: None (config-only option)
  # Common values: MNI152NLin2009cAsym, MNI152NLin6Asym, MNI305
  # Example: space: MNI152NLin2009cAsym
  space: MNI152NLin2009cAsym

# -------------------------------------------------------------------------------
# FILE PATTERN MATCHING
# -------------------------------------------------------------------------------
# Glob patterns for finding image files in derivatives folder(s).
# Use this for more flexible file selection than BIDS filters alone.

# Single pattern for finding all input images (useful for one-sample tests)
# CLI equivalent: --pattern / -p
# Examples:
#   file_pattern: "**/*_stat-effect_statmap.nii.gz"
#   file_pattern: "**/sub-*_task-nback_*.nii.gz"
#   file_pattern: "**/*GS*cvr*.nii.gz"
# Set to null to use BIDS filters only
file_pattern: null

# Pattern to exclude files after initial matching
# CLI equivalent: --exclude / -e
# Example: exclude_pattern: "*label-GM*"
# Set to null to not exclude any files
exclude_pattern: null

# Multiple patterns for two-sample or paired tests
# CLI equivalent: --patterns / -P (format: 'Name1=pattern1 Name2=pattern2')
# The sample names (keys) are used when specifying contrasts
# Examples:
#   For two-sample test:
#     sample_patterns:
#       Controls: "**/sub-*_group-control_*.nii.gz"
#       Patients: "**/sub-*_group-patient_*.nii.gz"
#   For method comparison:
#     sample_patterns:
#       GS: "*gas*GS*cvr*.nii.gz"
#       SSS: "*gas*SSS*cvr*.nii.gz"
# Set to null for one-sample or GLM analyses
sample_patterns: null

# -------------------------------------------------------------------------------
# DESIGN MATRIX (for GLM analysis)
# -------------------------------------------------------------------------------
# Specification for constructing the design matrix from participants.tsv
# CLI equivalent: None (config-only option, use GLM for complex designs)
design_matrix:
  # List of column names from participants.tsv to include as regressors
  # Examples:
  #   columns: []                           # Empty for simple t-tests
  #   columns: ["age", "sex"]               # Include age and sex
  #   columns: ["group", "age", "motion"]   # Multiple covariates
  columns: []

  # Whether to add an intercept column to the design matrix
  # Set to true for most analyses
  add_intercept: true

  # Specify which columns should be treated as categorical variables
  # Categorical variables are automatically dummy-coded
  # Examples:
  #   categorical_columns: null        # No categorical variables
  #   categorical_columns: ["sex"]     # Treat 'sex' as categorical
  #   categorical_columns: ["group", "site"]
  categorical_columns: null

  # Whether to standardize (z-score) continuous variables
  # Recommended for interpretability when mixing different scales
  standardize_continuous: true

# -------------------------------------------------------------------------------
# CONTRASTS
# -------------------------------------------------------------------------------
# Statistical contrasts to compute. The format depends on analysis type.
#
# For one-sample: Not needed (automatically tests mean vs zero)
# For two-sample with sample_patterns: Use sample names (e.g., "GS-SSS")
# For paired with sample_patterns: Use sample names (e.g., "Pre-Post")
# For GLM: Use regressor names from design matrix or participants.tsv
#
# CLI equivalent: --contrast / -C (single contrast only)
#
# Examples:
#   contrasts: []                              # No contrasts (for one-sample)
#   contrasts: ["GS-SSS"]                      # Method comparison
#   contrasts: ["Patients-Controls"]           # Group comparison
#   contrasts: ["age", "group1-group0"]        # GLM contrasts
#   contrasts: ["Pre-Post", "Post-Pre"]        # Both directions for paired test
contrasts: []

# -------------------------------------------------------------------------------
# GROUP COMPARISON SETTINGS (for two-sample tests)
# -------------------------------------------------------------------------------
# Configuration for independent two-sample t-tests comparing two groups
group_comparison:
  # Column name in participants.tsv that defines group membership
  # CLI equivalent: --group-column
  # Example: group_column: group
  # The column should contain exactly 2 unique values (e.g., "control", "patient")
  group_column: null

  # Explicitly specify the first group (optional)
  # CLI equivalent: None (config-only option)
  # If null, uses first unique value from group_column
  # Example: group1: control
  group1: null

  # Explicitly specify the second group (optional)
  # CLI equivalent: None (config-only option)
  # If null, uses second unique value from group_column
  # Example: group2: patient
  group2: null

# -------------------------------------------------------------------------------
# PAIRED TEST SETTINGS (for paired t-tests)
# -------------------------------------------------------------------------------
# Configuration for paired/within-subject comparisons
paired_test:
  # Column name that defines how to pair observations
  # CLI equivalent: --pair-by
  # Common values: "sub" (pair by subject), "session" (pair by session)
  # Example: pair_by: sub
  # Each unique value in this column represents one paired unit
  pair_by: null

  # METHOD 1: Using conditions (legacy approach)
  # First condition/session for paired comparison
  # CLI equivalent: --condition1
  # Example: condition1: pre
  condition1: null

  # Second condition/session for paired comparison
  # CLI equivalent: --condition2
  # Example: condition2: post
  condition2: null

  # METHOD 2: Using sample patterns (recommended)
  # Define patterns for each condition being compared
  # CLI equivalent: --patterns / -P
  # Example:
  #   sample_patterns:
  #     Pre: "*session-pre*.nii.gz"
  #     Post: "*session-post*.nii.gz"
  sample_patterns: null

# -------------------------------------------------------------------------------
# NORMALIZATION / SCALING
# -------------------------------------------------------------------------------
# Settings for normalizing/scaling input data before analysis
# Data is divided by the mean value within the mask.
#
# CLI equivalent: --scaling (format: 'key=pattern' or just 'pattern')
#
# NEW FORMAT (recommended): Specify both a key and pattern
#   The key appears in output filenames as '_scaling-{key}_'
#   Format: key=pattern
#   Examples:
#     --scaling 'brain=/path/*/func/*brain*mask.nii.gz'
#     --scaling 'gm=/path/*/anat/*GM*probseg.nii.gz'
#
# OLD FORMAT (backward compatible): Just the pattern (no key in outputs)
#   Format: pattern
#   Example: --scaling '/path/*/func/*brain*mask.nii.gz'
#
# Config file format:
#   scaling_key: brain  # Key for output filenames (appears as _scaling-brain_)
#   scaling_pattern: "/path/*/func/*brain*mask.nii.gz"  # Mask pattern
#
# Note: In the config file, you specify key and pattern separately.
# The mask pattern can be:
#   - A glob pattern with wildcards to find per-subject masks
#     Example: "/derivatives/fmriprep/sub-*/func/*brain*mask.nii.gz"
#   - A fixed path to use the same mask for all subjects
#     Example: "/path/to/template/MNI_brain_mask.nii.gz"
scaling_key: null

scaling_pattern: null

# -------------------------------------------------------------------------------
# STATISTICAL INFERENCE SETTINGS
# -------------------------------------------------------------------------------
# Configuration for thresholding and multiple comparison correction
#
# IMPORTANT: Two different significance levels (alpha) are used in neuroimaging:
#
# 1. alpha_uncorrected (default: 0.001) - "Cluster-forming threshold"
#    - Significance level for UNCORRECTED analysis
#    - Defines which voxels are considered "active" for cluster detection
#    - More stringent (smaller) values = fewer, more reliable clusters
#    - This is the p-value threshold applied to the uncorrected statistical map
#
# 2. alpha_corrected (default: 0.05) - "Significance level for corrected analyses"
#    - Used for FDR, Bonferroni, and permutation-based corrections
#    - Controls the family-wise error rate (FWER) or false discovery rate (FDR)
#    - Standard value is 0.05 (5% false positive rate after correction)
#
inference:
  # Significance level for CORRECTED thresholds (FDR, Bonferroni, permutation)
  # CLI equivalent: --alpha-corrected
  # Controls the error rate after multiple comparison correction
  # Typical values: 0.05 (standard), 0.01 (more conservative)
  alpha_corrected: 0.05

  # Significance level for UNCORRECTED analysis and cluster detection
  # CLI equivalent: --alpha-uncorrected
  # Also called "cluster-forming threshold" or "primary threshold"
  # Typical values: 0.001 (standard), 0.005, 0.01
  alpha_uncorrected: 0.001

  # Minimum cluster size in voxels
  # CLI equivalent: --cluster-threshold
  # Clusters smaller than this are discarded
  # Typical values: 10, 20, 50
  cluster_threshold: 10

  # Whether to use two-sided tests
  # CLI equivalent: None (config-only option)
  # true: Test for both positive and negative effects
  # false: Test for positive effects only (one-sided)
  two_sided: true

  # Multiple comparison correction methods to apply
  # CLI equivalent: None (config-only option, all methods run by default)
  # Valid options: "uncorrected", "fdr", "bonferroni", "permutation"
  # Results are saved for all specified methods
  # Example: corrections: ["uncorrected", "fdr", "bonferroni"]
  corrections:
    - uncorrected
    - fdr
    - bonferroni

  # Whether to run permutation testing for FWER correction
  # CLI equivalent: --permutation / --no-permutation
  # Permutation testing provides family-wise error rate control
  # Warning: Can be computationally expensive
  run_permutation: false

  # Number of permutations for permutation testing
  # CLI equivalent: --n-permutations
  # More permutations = more accurate but slower
  # Typical values: 1000, 5000, 10000
  # Only used if run_permutation is true
  n_permutations: 1000

# -------------------------------------------------------------------------------
# ATLAS FOR CLUSTER ANNOTATION
# -------------------------------------------------------------------------------
# Anatomical atlas used to label significant clusters
# CLI equivalent: --atlas
#
# Available atlases:
#   - "harvard_oxford": Harvard-Oxford cortical and subcortical atlases
#   - "aal": Automated Anatomical Labeling atlas
#   - "destrieux": Destrieux cortical atlas
#   - "schaefer": Schaefer cortical parcellations
#
# Example: atlas: harvard_oxford
atlas: harvard_oxford

# -------------------------------------------------------------------------------
# OUTPUT SETTINGS
# -------------------------------------------------------------------------------
# Control what outputs are generated
output:
  # Save statistical maps (z-maps, t-maps, effect size maps)
  # CLI equivalent: None (config-only option, always true in CLI)
  # Maps are saved as NIfTI files (.nii.gz)
  save_maps: true

  # Save cluster tables (CSV files with cluster locations and annotations)
  # CLI equivalent: None (config-only option, always true in CLI)
  # Tables include peak coordinates, cluster sizes, and anatomical labels
  save_tables: true

  # Generate HTML report with visualizations
  # CLI equivalent: --no-report (flag to disable)
  # Report includes glass brain plots, cluster tables, and analysis summary
  generate_report: true

  # Filename for HTML report
  # CLI equivalent: None (config-only option)
  # Saved in the output directory
  report_filename: report.html

# -------------------------------------------------------------------------------
# COMPUTATIONAL SETTINGS
# -------------------------------------------------------------------------------
# Number of parallel jobs for computation
# CLI equivalent: --n-jobs
# Values:
#   1: No parallelization (sequential)
#   -1: Use all available CPU cores
#   N: Use N cores
# Example: n_jobs: 4
n_jobs: 1

# Random seed for reproducibility
# CLI equivalent: None (config-only option)
# Set to an integer for reproducible permutation tests
# Example: random_state: 42
# Set to null for non-reproducible (random) behavior
random_state: null

# Verbosity level for logging
# CLI equivalent: --verbose / -v (count flag, use -v, -vv, -vvv)
# Values:
#   0: Warnings and errors only
#   1: Basic progress information (default)
#   2: Detailed information
#   3: Debug information
verbose: 1

# ===============================================================================
# EXAMPLES FOR COMMON ANALYSES
# ===============================================================================
#
# --- ONE-SAMPLE T-TEST ---
# Test if mean activation is significantly different from zero
#
# analysis_type: one-sample
# file_pattern: "**/*_stat-effect_statmap.nii.gz"
# bids_filters:
#   task: nback
#
# --- TWO-SAMPLE T-TEST (GROUP COMPARISON) ---
# Compare two independent groups using participants.tsv
#
# analysis_type: two-sample
# group_comparison:
#   group_column: group
# contrasts: ["Patients-Controls"]
#
# --- TWO-SAMPLE T-TEST (METHOD COMPARISON) ---
# Compare two methods using file patterns
#
# analysis_type: two-sample
# sample_patterns:
#   GS: "*gas*GS*cvr*.nii.gz"
#   SSS: "*gas*SSS*cvr*.nii.gz"
# contrasts: ["GS-SSS"]
#
# --- PAIRED T-TEST (SESSION COMPARISON) ---
# Compare within-subject across sessions
#
# analysis_type: paired
# paired_test:
#   pair_by: sub
#   sample_patterns:
#     Pre: "*session-pre*.nii.gz"
#     Post: "*session-post*.nii.gz"
# contrasts: ["Post-Pre"]
#
# --- GLM WITH COVARIATES ---
# Custom design matrix with age and sex as covariates
#
# analysis_type: glm
# design_matrix:
#   columns: ["age", "sex"]
#   categorical_columns: ["sex"]
# contrasts: ["age"]
#
# ===============================================================================
"""

    # Write the template to file
    with open(output_path, 'w') as f:
        f.write(template)

    logger.info(f"Configuration file created: {output_path}")
    return output_path
