"""
Command-line interface for StatCraft.

This module provides the CLI entry point for running
second-level neuroimaging analyses on BIDS-compliant datasets.
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

from statcraft import __version__
from statcraft.config import Config, create_default_config
from statcraft.pipeline import StatCraftPipeline

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ColoredHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter with colored section headers."""

    def __init__(self, prog, indent_increment=2, max_help_position=40, width=100):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = f'{Colors.BOLD}Usage:{Colors.END} '
        return super()._format_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        if heading:
            heading = f'{Colors.BOLD}{Colors.CYAN}{heading}{Colors.END}'
        super().start_section(heading)


def _validate_bids(bids_dir: str, verbose: int = 0) -> None:
    """
    Validate BIDS dataset structure.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS rawdata directory.
    verbose : int
        Verbosity level.
        
    Raises
    ------
    ValueError
        If BIDS validation fails.
    """
    from statcraft.core.data_loader import DataLoader
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = DataLoader(
            bids_dir=bids_dir,
            derivatives=[bids_dir],
            output_dir=tmpdir,
        )
        
        # Basic validation: check if participants exist
        if loader.participants is None or len(loader.participants) == 0:
            raise ValueError("No participants found in BIDS dataset")
        
        if verbose > 0:
            print(f"  Found {len(loader.participants)} participants")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""

    description = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     StatCraft v{__version__:<58}║
    ║                  BIDS-based Second-Level Analysis Tool                    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

    {Colors.BOLD}Description:{Colors.END}
      StatCraft performs second-level neuroimaging analyses on BIDS-compliant datasets.
      It processes individual-level statistics and applies group-level workflows to
      produce analysis-ready outputs that conform to BIDS derivative standards.

    {Colors.BOLD}Supported Analyses:{Colors.END}
      • One-sample t-tests
      • Two-sample t-tests (group comparisons or method comparisons)
      • Paired t-tests (within-subject comparisons)
      • General Linear Model (GLM) with continuous/categorical regressors
      • Connectivity matrix analysis (edge-wise statistics)

    {Colors.BOLD}Workflow:{Colors.END}
      1. Discover input data from BIDS dataset structure
      2. Validate data integrity and consistency
      3. Configure analysis parameters
      4. Execute main processing pipeline
      5. Generate BIDS-compliant outputs with metadata
      6. Produce analysis reports and quality metrics
    """)

    epilog = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}EXAMPLES{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

    {Colors.BOLD}Configuration File:{Colors.END}

      {Colors.YELLOW}# Generate default configuration{Colors.END}
      statcraft --init-config config.yaml

      {Colors.YELLOW}# Run analysis with config file{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -c config.yaml

    {Colors.BOLD}One-Sample Tests:{Colors.END}

      {Colors.YELLOW}# Process all subjects with default pattern{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample

      {Colors.YELLOW}# Use specific glob pattern for images{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample -p '*gas*cvr*.nii.gz'

      {Colors.YELLOW}# Exclude specific images from analysis{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample -p '*cvr*.nii.gz' -e '*label-bad*'

    {Colors.BOLD}Group Comparisons (Two-Sample):{Colors.END}

      {Colors.YELLOW}# Using group column in participants.tsv{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t two-sample --group-column group

      {Colors.YELLOW}# Using sample-specific patterns{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t two-sample \\
          -P 'GS=*gas*GS*cvr*.nii.gz SSS=*gas*SSS*cvr*.nii.gz' -C 'GS-SSS'

    {Colors.BOLD}Paired Comparisons:{Colors.END}

      {Colors.YELLOW}# Within-subject pairing by subject ID{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t paired --pair-by sub \\
          -P 'pre=*pre*.nii.gz post=*post*.nii.gz' -C 'post-pre'

    {Colors.BOLD}General Linear Model (GLM):{Colors.END}

      {Colors.YELLOW}# Age effect on connectivity{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives --data-type connectivity \\
          -t glm --regressors age -C age -p '**/*_connmat.npy'

      {Colors.YELLOW}# Sex effect (categorical) controlling for age{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t glm \\
          --regressors age sex --categorical-regressors sex -C 'sex_M-sex_F'

    {Colors.BOLD}BIDS Entity Filtering:{Colors.END}

      {Colors.YELLOW}# Process specific participants{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample -p participant 01 02 03

      {Colors.YELLOW}# Process specific task and session{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample --task rest --session 01

      {Colors.YELLOW}# Filter by template space{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample --space MNI152NLin2009cAsym

    {Colors.BOLD}Temporal Processing:{Colors.END}

      {Colors.YELLOW}# Drop initial volumes and set minimum segment length{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample \\
          --drop-initial 4 --min-segment-length 5

    {Colors.BOLD}Advanced Options:{Colors.END}

      {Colors.YELLOW}# Enable permutation testing{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample --permutation

      {Colors.YELLOW}# Custom atlas and cluster analysis{Colors.END}
      statcraft /data/bids /data/output participant \\
          -d /data/derivatives/fmriprep -t one-sample \\
          --atlas aal --extra-cluster-analysis

    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}MORE INFORMATION{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

      Documentation:  https://github.com/arovai/StatCraft
      Report Issues:  https://github.com/arovai/StatCraft/issues
      Version:        {__version__}
    """)

    parser = argparse.ArgumentParser(
        prog="statcraft",
        description=description,
        epilog=epilog,
        formatter_class=ColoredHelpFormatter,
        add_help=False,
    )

    # =========================================================================
    # REQUIRED ARGUMENTS
    # =========================================================================
    required = parser.add_argument_group(
        f'{Colors.BOLD}Required Arguments{Colors.END}'
    )

    required.add_argument(
        "input_dir",
        type=Path,
        metavar="INPUT_DIR",
        help="Path to the BIDS dataset root directory.",
    )

    required.add_argument(
        "output_dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help="Path to output directory for analysis derivatives.",
    )

    required.add_argument(
        "analysis_level",
        choices=["participant"],
        metavar="{participant}",
        help="Analysis level. Currently only 'participant' is supported.",
    )

    # =========================================================================
    # GENERAL OPTIONS
    # =========================================================================
    general = parser.add_argument_group(
        f'{Colors.BOLD}General Options{Colors.END}'
    )

    general.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )

    general.add_argument(
        "--version",
        action="version",
        version=f"statcraft {__version__}",
        help="Show program version and exit.",
    )

    general.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Enable verbose output (can be specified multiple times).",
    )

    general.add_argument(
        "-c", "--config",
        type=Path,
        metavar="FILE",
        help="Path to configuration file (.json, .yaml, or .yml). "
             "CLI arguments override config file settings.",
    )

    general.add_argument(
        "--init-config",
        type=Path,
        metavar="FILE",
        help="Generate a default configuration file and exit.",
    )

    # =========================================================================
    # DERIVATIVES OPTIONS
    # =========================================================================
    derivatives = parser.add_argument_group(
        f'{Colors.BOLD}Input Derivatives Options{Colors.END}'
    )

    derivatives.add_argument(
        "-d", "--derivatives",
        action="append",
        metavar="PATH",
        dest="derivatives",
        help="Path to BIDS derivatives folder (e.g., fmriprep output). "
             "Can be specified multiple times.",
    )

    # =========================================================================
    # BIDS ENTITY FILTERS
    # =========================================================================
    filters = parser.add_argument_group(
        f'{Colors.BOLD}BIDS Entity Filters{Colors.END}',
        "Filter which data to process based on BIDS entities."
    )

    filters.add_argument(
        "-p", "--participant-label",
        metavar="LABEL",
        dest="participant_label",
        nargs='+',
        help="Process one or more participants (without 'sub-' prefix).",
    )

    filters.add_argument(
        "-t", "--task",
        metavar="TASK",
        help="Process only this task (without 'task-' prefix).",
    )

    filters.add_argument(
        "-s", "--session",
        metavar="SESSION",
        help="Process only this session (without 'ses-' prefix).",
    )

    filters.add_argument(
        "-r", "--run",
        metavar="RUN",
        type=int,
        help="Process only this run number.",
    )

    filters.add_argument(
        "--space",
        metavar="SPACE",
        help="Process only data in this template space "
             "(e.g., 'MNI152NLin2009cAsym').",
    )

    filters.add_argument(
        "--label",
        metavar="STRING",
        help="Custom label added to all output filenames (BIDS entity).",
    )

    # =========================================================================
    # ANALYSIS OPTIONS
    # =========================================================================
    analysis = parser.add_argument_group(
        f'{Colors.BOLD}Analysis Options{Colors.END}'
    )

    analysis.add_argument(
        "--analysis-type",
        metavar="TYPE",
        choices=["one-sample", "two-sample", "paired", "glm"],
        dest="analysis_type",
        help="Type of analysis to run.",
    )

    analysis.add_argument(
        "--data-type",
        metavar="TYPE",
        choices=["auto", "nifti", "connectivity"],
        default="auto",
        dest="data_type",
        help="Type of input data. Auto-detects from extension (.nii.gz for NIfTI, .npy for connectivity). "
             "(default: auto)",
    )

    analysis.add_argument(
        "--contrast", "-C",
        metavar="EXPR",
        help="Contrast expression for the analysis. For GLM: use design matrix column names "
             "(e.g., 'age'). For comparisons with patterns: use sample names (e.g., 'GS-SSS').",
    )

    # =========================================================================
    # PATTERN/FILE OPTIONS
    # =========================================================================
    patterns_group = parser.add_argument_group(
        f'{Colors.BOLD}Pattern Options{Colors.END}',
        "Specify how to find and process input files."
    )

    patterns_group.add_argument(
        "--pattern",
        metavar="GLOB",
        help="Glob pattern for finding image files (e.g., '*.nii.gz' or '**/*GS*cvr*.nii.gz'). "
             "Use quotes to prevent shell expansion.",
    )

    patterns_group.add_argument(
        "--patterns",
        metavar="NAME=GLOB ...",
        help="Multiple patterns for comparisons. Format: 'Name1=pattern1 Name2=pattern2' "
             "(e.g., 'GS=*GS*cvr*.nii.gz SSS=*SSS*cvr*.nii.gz'). Sample names used in --contrast.",
    )

    patterns_group.add_argument(
        "--exclude",
        metavar="GLOB",
        help="Glob pattern to exclude files from matching. Applied after --pattern. "
             "For label-specific excludes: 'LABEL1=pattern1 LABEL2=pattern2'.",
    )

    # =========================================================================
    # DESIGN MATRIX / REGRESSION OPTIONS
    # =========================================================================
    design = parser.add_argument_group(
        f'{Colors.BOLD}Design Matrix Options{Colors.END}',
        "Configure regressors and group comparisons."
    )

    design.add_argument(
        "--group-column",
        metavar="COLUMN",
        dest="group_column",
        help="Column name in participants.tsv for group comparison (two-sample test).",
    )

    design.add_argument(
        "--regressors",
        metavar="COLUMN",
        nargs='+',
        help="Regressor column names from participants.tsv (e.g., age sex IQ). "
             "Continuous variables are z-scored; categorical variables are dummy-coded.",
    )

    design.add_argument(
        "--categorical-regressors",
        metavar="COLUMN",
        nargs='+',
        dest="categorical_regressors",
        help="Columns to treat as categorical (dummy-coded).",
    )

    design.add_argument(
        "--no-standardize-regressors",
        action="store_true",
        dest="no_standardize_regressors",
        help="Disable z-scoring of continuous regressors (keep original units).",
    )

    # =========================================================================
    # PAIRED TEST OPTIONS
    # =========================================================================
    paired = parser.add_argument_group(
        f'{Colors.BOLD}Paired Test Options{Colors.END}',
        "Configure paired t-test analysis."
    )

    paired.add_argument(
        "--pair-by",
        metavar="COLUMN",
        dest="pair_by",
        help="Column name for pairing subjects in paired t-test.",
    )

    paired.add_argument(
        "--condition1",
        metavar="VALUE",
        dest="condition1",
        help="First condition for paired comparison.",
    )

    paired.add_argument(
        "--condition2",
        metavar="VALUE",
        dest="condition2",
        help="Second condition for paired comparison.",
    )

    # =========================================================================
    # NORMALIZATION / SCALING OPTIONS
    # =========================================================================
    normalization = parser.add_argument_group(
        f'{Colors.BOLD}Normalization Options{Colors.END}',
        "Options for data normalization."
    )

    normalization.add_argument(
        "--scaling",
        metavar="KEY=PATTERN",
        help="Data scaling/normalization using a mask. Format: 'key=pattern' "
             "(e.g., 'brain=/path/*/func/*brain*mask.nii.gz'). Data is divided by mean within mask. "
             "Mutually exclusive with --zscore.",
    )

    normalization.add_argument(
        "--zscore",
        action="store_true",
        help="Z-score individual data at subject level: (x - mean(x)) / std(x). "
             "Standardizes each subject to mean=0, std=1. Requires --mask.",
    )

    normalization.add_argument(
        "--mask",
        metavar="PATTERN",
        help="Brain mask pattern for z-scoring (required with --zscore). "
             "Format: '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz'.",
    )

    # =========================================================================
    # TEMPORAL PROCESSING OPTIONS
    # =========================================================================
    temporal = parser.add_argument_group(
        f'{Colors.BOLD}Temporal Processing Options{Colors.END}',
        "Options for temporal volume selection and segmentation."
    )

    temporal.add_argument(
        "--threshold",
        metavar="VALUE",
        type=float,
        dest="threshold",
        help="Quality threshold for volume selection. Volumes below threshold excluded.",
    )

    temporal.add_argument(
        "--extend",
        metavar="N",
        type=int,
        default=0,
        help="Extend exclusion to N volumes before AND after flagged volumes (default: 0).",
    )

    temporal.add_argument(
        "--min-segment-length",
        metavar="N",
        type=int,
        dest="min_segment_length",
        default=0,
        help="Minimum contiguous segment length to retain after exclusion. Requires --threshold (default: 0).",
    )

    temporal.add_argument(
        "--drop-initial",
        metavar="N",
        type=int,
        dest="drop_initial",
        default=0,
        help="Number of initial volumes to drop (default: 0).",
    )

    # =========================================================================
    # INFERENCE / STATISTICS OPTIONS
    # =========================================================================
    inference = parser.add_argument_group(
        f'{Colors.BOLD}Inference Options{Colors.END}',
        "Statistical testing parameters."
    )

    inference.add_argument(
        "--alpha-corrected",
        metavar="VALUE",
        type=float,
        default=0.05,
        dest="alpha_corrected",
        help="Significance level (α) for CORRECTED thresholds after multiple comparison correction "
             "(default: 0.05).",
    )

    inference.add_argument(
        "--alpha-uncorrected",
        metavar="VALUE",
        type=float,
        default=0.001,
        dest="alpha_uncorrected",
        help="Significance level (α) for UNCORRECTED analysis (cluster-forming threshold) "
             "(default: 0.001).",
    )

    inference.add_argument(
        "--cluster-threshold",
        metavar="N",
        type=int,
        default=10,
        dest="cluster_threshold",
        help="Minimum cluster size in voxels (default: 10).",
    )

    inference.add_argument(
        "--permutation",
        action="store_true",
        help="Run permutation testing for FWER correction.",
    )

    inference.add_argument(
        "--n-permutations",
        metavar="N",
        type=int,
        default=5000,
        dest="n_permutations",
        help="Number of permutations for permutation testing (default: 5000).",
    )

    # =========================================================================
    # PROCESSING / OUTPUT OPTIONS
    # =========================================================================
    processing = parser.add_argument_group(
        f'{Colors.BOLD}Processing Options{Colors.END}'
    )

    processing.add_argument(
        "--atlas",
        metavar="NAME",
        default="harvard_oxford",
        help="Atlas for anatomical annotation (default: harvard_oxford).",
    )

    processing.add_argument(
        "--n-jobs",
        metavar="N",
        type=int,
        default=1,
        dest="n_jobs",
        help="Number of parallel jobs (-1 for all cores) (default: 1).",
    )

    processing.add_argument(
        "--smoothing-fwhm",
        metavar="MM",
        type=float,
        default=5.0,
        dest="smoothing_fwhm",
        help="Smoothing kernel FWHM in mm for second-level GLM. Set to 0 to disable (default: 5.0).",
    )

    processing.add_argument(
        "--no-report",
        action="store_true",
        dest="no_report",
        help="Disable HTML report generation.",
    )

    processing.add_argument(
        "--skip-bids-validator",
        action="store_true",
        dest="skip_bids_validator",
        help="Skip BIDS validation of the rawdata folder.",
    )

    # =========================================================================
    # CLUSTER ANALYSIS OPTIONS
    # =========================================================================
    cluster = parser.add_argument_group(
        f'{Colors.BOLD}Cluster Analysis Options{Colors.END}'
    )

    cluster.add_argument(
        "--extra-cluster-analysis",
        action="store_true",
        dest="extra_cluster_analysis",
        help="Perform detailed cluster analysis including atlas overlap and tissue statistics.",
    )

    cluster.add_argument(
        "--cluster-overlap-threshold",
        metavar="PERCENT",
        type=float,
        default=5.0,
        dest="cluster_overlap_threshold",
        help="Minimum percentage threshold for reporting atlas region overlaps (default: 5.0).",
    )

    # =========================================================================
    # OUTPUT / DATA OPTIONS
    # =========================================================================
    output = parser.add_argument_group(
        f'{Colors.BOLD}Output Options{Colors.END}'
    )

    output.add_argument(
        "--save-supplementary-data",
        action="store_true",
        dest="save_supplementary_data",
        help="Save supplementary data (original, normalized, masks, etc.). Disabled by default.",
    )

    return parser


def parse_derivatives_arg(derivatives_list: list) -> dict:
    """
    Parse derivatives arguments into dictionary.
    
    Expected format: name=path or just path
    """
    if not derivatives_list:
        return {}

    derivatives_dict = {}
    for derivative_arg in derivatives_list:
        if "=" in derivative_arg:
            name, path = derivative_arg.split("=", 1)
            derivatives_dict[name] = Path(path)
        else:
            # Just a path, use a default name
            derivatives_dict[f"derivatives_{len(derivatives_dict)}"] = Path(derivative_arg)

    return derivatives_dict


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --init-config flag
    if args.init_config:
        output_path = Path(args.init_config)
        # Add appropriate extension if missing
        if not output_path.suffix:
            output_path = output_path.with_suffix(".yaml")
        create_default_config(output_path)
        print(f"{Colors.GREEN}✓ Configuration file created: {output_path}{Colors.END}")
        return
    
    # Set up logging
    log_level = logging.WARNING - (args.verbose * 10)
    logging.basicConfig(
        level=max(log_level, logging.DEBUG),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"{Colors.BOLD}{Colors.GREEN}StatCraft v{__version__}{Colors.END}")
    print("=" * 40)
    
    # Load configuration first to check if paths are in config
    cfg = None
    if args.config:
        try:
            cfg = Config(config_file=args.config)
            if args.verbose > 0:
                print(f"Using configuration from: {args.config}")
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to load config file: {e}{Colors.END}", file=sys.stderr)
            sys.exit(1)

    # Determine input and output directories
    bids_dir = args.input_dir
    output_dir = args.output_dir
    
    # Run BIDS validation unless skipped
    if not args.skip_bids_validator:
        print("Validating BIDS dataset...")
        try:
            _validate_bids(str(bids_dir), args.verbose)
            print(f"{Colors.GREEN}✓ BIDS validation passed{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}✗ BIDS validation failed: {str(e)}{Colors.END}", file=sys.stderr)
            print("Use --skip-bids-validator to bypass this check.", file=sys.stderr)
            sys.exit(1)

    # Validate mutual exclusivity of --scaling and --zscore
    if args.scaling and args.zscore:
        print(f"{Colors.RED}✗ Error: --scaling and --zscore are mutually exclusive.{Colors.END}", file=sys.stderr)
        print("Choose one method to handle inter-individual differences:", file=sys.stderr)
        print("  --scaling: Normalize by dividing by mean within a mask", file=sys.stderr)
        print("  --zscore: Standardize to mean=0, std=1", file=sys.stderr)
        sys.exit(1)

    # Validate that --mask is required with --zscore
    if args.zscore and not args.mask:
        print(f"{Colors.RED}✗ Error: --mask is required when using --zscore.{Colors.END}", file=sys.stderr)
        print("Provide a brain mask pattern, e.g.:", file=sys.stderr)
        print("  --mask '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz'", file=sys.stderr)
        sys.exit(1)

    # Parse multi-sample patterns if provided
    sample_patterns = None
    if args.patterns:
        if args.verbose > 0:
            print(f"Parsing sample patterns: {args.patterns}")
        sample_patterns = {}
        # Split by whitespace and parse Name=pattern pairs
        parts = args.patterns.split()
        for part in parts:
            if '=' not in part:
                print(f"{Colors.RED}✗ Invalid pattern format: {part}{Colors.END}", file=sys.stderr)
                print("Expected format: 'Name1=pattern1 Name2=pattern2'", file=sys.stderr)
                sys.exit(1)
            name, patt = part.split('=', 1)
            sample_patterns[name] = patt
            if args.verbose > 0:
                print(f"  Sample '{name}': {patt}")

        if len(sample_patterns) < 2:
            print(f"{Colors.RED}✗ At least two samples required for --patterns{Colors.END}", file=sys.stderr)
            sys.exit(1)

    # Parse exclude patterns when using multi-sample patterns
    exclude_patterns_dict = None
    if sample_patterns and args.exclude:
        # Check if exclude contains '=' indicating label-specific patterns
        if '=' in args.exclude:
            if args.verbose > 0:
                print(f"Parsing label-specific exclude patterns: {args.exclude}")
            exclude_patterns_dict = {}
            # Split by whitespace and parse Label=pattern pairs
            parts = args.exclude.split()
            for part in parts:
                if '=' not in part:
                    print(f"{Colors.RED}✗ Invalid exclude pattern format: {part}{Colors.END}", file=sys.stderr)
                    print("Expected format: 'Label1=pattern1 Label2=pattern2'", file=sys.stderr)
                    sys.exit(1)
                label, patt = part.split('=', 1)
                # Validate that label exists in sample_patterns
                if label not in sample_patterns:
                    print(f"{Colors.RED}✗ Exclude pattern label '{label}' not found in --patterns{Colors.END}", file=sys.stderr)
                    print(f"   Available labels: {', '.join(sample_patterns.keys())}", file=sys.stderr)
                    sys.exit(1)
                exclude_patterns_dict[label] = patt
                if args.verbose > 0:
                    print(f"  Exclude for '{label}': {patt}")

    # Parse scaling option (key=value format)
    scaling_key = None
    scaling_pattern = None
    if args.scaling:
        if '=' in args.scaling:
            # New format: key=pattern
            parts = args.scaling.split('=', 1)
            if len(parts) == 2:
                scaling_key, scaling_pattern = parts
                if args.verbose > 0:
                    print(f"Scaling: '{scaling_key}' using pattern: {scaling_pattern}")
            else:
                print(f"{Colors.RED}✗ Invalid scaling format: {args.scaling}{Colors.END}", file=sys.stderr)
                print("Expected format: 'key=pattern' (e.g., 'brain=/path/to/mask.nii.gz')", file=sys.stderr)
                sys.exit(1)
        else:
            # Backward compatibility: just pattern (no key)
            scaling_pattern = args.scaling
            scaling_key = None
            print(f"{Colors.YELLOW}⚠ Warning: Using scaling without key. Consider using 'key=pattern' format{Colors.END}", file=sys.stderr)
            print(f"   Example: --scaling 'brain={args.scaling}'", file=sys.stderr)

    # Warn if --task/--session/--participant-label are used with --pattern
    if args.pattern and (args.task or args.session or args.participant_label):
        print(f"{Colors.YELLOW}⚠ Warning: When using --pattern, BIDS filters are applied AFTER pattern matching.{Colors.END}", file=sys.stderr)
        print("   It's better to include filters directly in the pattern.", file=sys.stderr)

    # Build configuration from CLI options
    config_overrides = {}

    if args.analysis_type:
        config_overrides["analysis_type"] = args.analysis_type
    if args.contrast:
        config_overrides["contrast"] = args.contrast

    # BIDS filters should be nested under bids_filters key
    if args.task or args.session or args.participant_label:
        config_overrides["bids_filters"] = {}
        if args.task:
            config_overrides["bids_filters"]["task"] = args.task
        if args.session:
            config_overrides["bids_filters"]["session"] = args.session
        if args.participant_label:
            config_overrides["bids_filters"]["participant"] = list(args.participant_label)
    
    # Group comparison
    if args.group_column:
        config_overrides["group_comparison"] = {"group_column": args.group_column}
    
    # Design matrix with regressors
    if args.regressors:
        config_overrides["design_matrix"] = {
            "columns": list(args.regressors),
            "add_intercept": True,
            "standardize_continuous": not args.no_standardize_regressors,
        }
        if args.categorical_regressors:
            config_overrides["design_matrix"]["categorical_columns"] = list(args.categorical_regressors)
        
        if args.verbose > 0:
            print(f"Design matrix configuration:")
            print(f"  Regressors: {', '.join(args.regressors)}")
            if args.categorical_regressors:
                print(f"  Categorical: {', '.join(args.categorical_regressors)}")
            if args.no_standardize_regressors:
                print(f"  Standardize continuous: disabled")
    
    # Paired test
    if args.pair_by:
        config_overrides["paired_test"] = {
            "pair_by": args.pair_by,
            "condition1": args.condition1,
            "condition2": args.condition2,
        }
        # Add sample patterns if provided
        if sample_patterns:
            config_overrides["paired_test"]["sample_patterns"] = sample_patterns

    # Two-sample test with patterns
    if sample_patterns and args.analysis_type == "two-sample":
        config_overrides["sample_patterns"] = sample_patterns
    
    # Inference settings
    config_overrides["inference"] = {
        "alpha_corrected": args.alpha_corrected,
        "alpha_uncorrected": args.alpha_uncorrected,
        "cluster_threshold": args.cluster_threshold,
        "run_permutation": args.permutation,
        "n_permutations": args.n_permutations,
    }
    
    config_overrides["atlas"] = args.atlas
    config_overrides["n_jobs"] = args.n_jobs
    config_overrides["verbose"] = args.verbose

    # GLM settings
    config_overrides["glm"] = {
        "smoothing_fwhm": args.smoothing_fwhm if args.smoothing_fwhm > 0 else None,
    }

    # Cluster analysis settings
    config_overrides["extra_cluster_analysis"] = args.extra_cluster_analysis
    config_overrides["cluster_overlap_threshold"] = args.cluster_overlap_threshold

    # Normalization settings
    if args.zscore:
        config_overrides["zscore"] = True
        config_overrides["mask_pattern"] = args.mask
    if scaling_pattern:
        config_overrides["scaling_pattern"] = scaling_pattern
        if scaling_key:
            config_overrides["scaling_key"] = scaling_key

    # Output settings
    config_overrides["output"] = {
        "generate_report": not args.no_report,
        "save_supplementary_data": args.save_supplementary_data,
    }

    # Add data type to config
    config_overrides["data_type"] = args.data_type
    
    try:
        # Load or update configuration
        if cfg:
            # cfg was already loaded from config file
            # Now apply the overrides
            cfg._update_nested(cfg.data, config_overrides)
        else:
            cfg = Config(**config_overrides)
        
        # Create and run pipeline
        pipeline = StatCraftPipeline(
            bids_dir=str(bids_dir),
            output_dir=str(output_dir),
            derivatives=[str(d) for d in (args.derivatives or [])],
            config=cfg,
        )

        # Determine if we should run connectivity analysis
        run_connectivity = False
        if args.data_type == "connectivity":
            run_connectivity = True
        elif args.data_type == "auto" and args.pattern and ".npy" in args.pattern:
            run_connectivity = True
            print("Auto-detected connectivity matrix analysis from .npy pattern")

        if run_connectivity:
            # Run connectivity analysis
            results = pipeline.run_connectivity_analysis(
                pattern=args.pattern,
                exclude_pattern=exclude_patterns_dict if exclude_patterns_dict else args.exclude,
                sample_patterns=sample_patterns,
            )
        else:
            # Run standard NIfTI analysis
            results = pipeline.run(
                pattern=args.pattern,
                exclude_pattern=exclude_patterns_dict if exclude_patterns_dict else args.exclude,
                sample_patterns=sample_patterns,
                scaling=scaling_pattern,
                scaling_key=scaling_key,
                zscore=args.zscore,
                mask=args.mask,
            )
        
        print(f"\n{Colors.GREEN}✓ Analysis completed successfully!{Colors.END}")
        print(f"  Results saved to: {output_dir}")
        
        if "report" in results.get("saved_files", {}):
            print(f"  Report: {results['saved_files']['report']}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logger.exception("Analysis failed")
        print(f"\n{Colors.RED}✗ Analysis failed: {str(e)}{Colors.END}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
