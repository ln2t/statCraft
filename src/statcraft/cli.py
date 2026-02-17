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
from typing import Any, Dict, Optional

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


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""

    description = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     StatCraft v{__version__:<46}║
    ║                     Second-Level Neuroimaging Analysis                       ║
    ╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

    {Colors.BOLD}Description:{Colors.END}
      StatCraft performs second-level neuroimaging analyses using flexible pattern 
      matching for file discovery. It processes individual-level statistics and applies 
      group-level workflows to produce analysis-ready outputs.

    {Colors.BOLD}Supported Analyses:{Colors.END}
      • One-sample t-tests
      • Two-sample t-tests (group comparisons or method comparisons)
      • Paired t-tests (within-subject comparisons)
      • General Linear Model (GLM) with continuous/categorical regressors
      • Connectivity matrix analysis (edge-wise statistics)

    {Colors.BOLD}Workflow:{Colors.END}
      1. Discover input data using flexible glob patterns
      2. Filter participants (optionally using --participant-label)
      3. Validate data integrity and consistency
      4. Configure analysis parameters
      5. Execute main processing pipeline
      6. Generate outputs with metadata and analysis reports
    """)

    epilog = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}EXAMPLES{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

    {Colors.BOLD}Generate Default Configuration:{Colors.END}

      {Colors.YELLOW}statcraft --init-config config.yaml{Colors.END}

    {Colors.BOLD}One-Sample T-Tests:{Colors.END}

      {Colors.YELLOW}# Simple one-sample t-test on motor task maps{Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type one-sample \\
          --pattern "*task-motor*beta1*.nii.gz"

    {Colors.BOLD}Two-Sample T-Tests (Group Comparisons):{Colors.END}

      {Colors.YELLOW}# Compare controls vs patients using distinguishable names{Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type two-sample \\
          --patterns "controls=sub-c*.nii.gz patients=sub-p*.nii.gz"

    {Colors.BOLD}Paired T-Tests (Within-Subject Comparisons):{Colors.END}

      {Colors.YELLOW}# Compare session 1 to session 2, paired by subject (default pairing){Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type paired \\
          --patterns "session1=*ses-1*.nii.gz session2=*ses-2*.nii.gz"

      {Colors.YELLOW}# Pair by session instead of subject{Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type paired \\
          --patterns "pre=*ses-pre*.nii.gz post=*ses-post*.nii.gz" \\
          --pair-by "ses"

    {Colors.BOLD}General Linear Model (GLM):{Colors.END}

      {Colors.YELLOW}# GLM with all columns from participants.tsv (default behavior){Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type glm \\
          --participants-file /path/to/participants.tsv \\
          --categorical-regressors sex group \\
          --pattern "*stat-effect*.nii.gz"

      {Colors.YELLOW}# GLM with specific regressors and multiple contrasts{Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type glm \\
          --participants-file /path/to/participants.tsv \\
          --regressors age IQ \\
          --categorical-regressors sex \\
          --contrasts age M-F "0.5*M+0.5*F-mean" \\
          --pattern "*stat-effect*.nii.gz"

      {Colors.YELLOW}# GLM with no intercept and selective z-scoring{Colors.END}
      statcraft /path/to/first_level /path/to/output group \\
          --analysis-type glm \\
          --participants-file /path/to/participants.tsv \\
          --regressors sex age IQ \\
          --categorical-regressors sex \\
          --no-standardize-regressors age \\
          --no-intercept \\
          --contrasts age M-F \\
          --pattern "*stat-effect*.nii.gz"

    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}MORE INFORMATION{Colors.END}
    {Colors.GREEN}═════════════════════════════════════════════════════════════════════════════════{Colors.END}

    Documentation:  https://github.com/ln2t/StatCraft
    Report Issues:  https://github.com/ln2t/StatCraft/issues
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
        help="Path to dataset root or derivatives folder. Can be either:\n"
             "  • Dataset root directory (use with --derivatives for separate derivatives)\n"
             "  • Derivatives folder directly (optional --participants-file if needed)",
    )

    required.add_argument(
        "output_dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help="Path to output directory for analysis derivatives.",
    )

    required.add_argument(
        "analysis_level",
        choices=["group"],
        metavar="{group}",
        help="Analysis level. Currently only 'group' is supported.",
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

    derivatives.add_argument(
        "--participants-file",
        type=Path,
        metavar="FILE",
        dest="participants_file",
        help="Path to participants.tsv file. Use this when INPUT_DIR is directly a derivatives "
             "folder without participants.tsv. If not provided, will look for participants.tsv in INPUT_DIR.",
    )

    # =========================================================================
    # PARTICIPANT FILTERING
    # =========================================================================
    filters = parser.add_argument_group(
        f'{Colors.BOLD}Participant Filtering{Colors.END}',
        "Filter which participants to include in the analysis."
    )

    filters.add_argument(
        "-p", "--participant-label",
        metavar="LABEL",
        dest="participant_label",
        nargs='+',
        help="Process one or more participants (without 'sub-' prefix). "
             "If not specified, all participants found will be included. "
             "Use glob patterns for other filtering (e.g., '*task-rest*' in --pattern).",
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
        help="Type of analysis to run. Choices: 'one-sample' (one-sample t-test), "
             "'two-sample' (group comparison or method comparison), 'paired' (within-subject comparison), "
             "'glm' (General Linear Model with continuous/categorical regressors).",
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
        "--contrasts", "-C",
        metavar="EXPR",
        nargs='+',
        help="Contrast expression(s) for the analysis. Can specify multiple contrasts separated by spaces. "
             "For GLM: use design matrix column names (e.g., 'age', 'sex_M-sex_F'), "
             "original categorical values (e.g., 'M-F'), or 'mean' to refer to the intercept. "
             "Examples: --contrasts age M-F '0.5*M+0.5*F-mean'. "
             "For comparisons with patterns: use sample names (e.g., 'GS-SSS').",
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
        "Configure regressors for GLM analysis."
    )

    design.add_argument(
        "--regressors",
        metavar="COLUMN",
        nargs='+',
        help="Regressor column names from participants.tsv (e.g., age sex IQ). "
             "If not specified, ALL columns from participants.tsv are used by default. "
             "All regressors are treated as continuous and z-scored by default. "
             "Use --categorical-regressors to treat specific columns as categorical (dummy-coded).",
    )

    design.add_argument(
        "--categorical-regressors",
        metavar="COLUMN",
        nargs='+',
        dest="categorical_regressors",
        help="Columns to treat as categorical (dummy-coded) in GLM. "
             "Can include columns NOT specified in --regressors (they will be fetched from participants.tsv). "
             "Use for columns with non-numerical values (e.g., sex with M/F, treatment with control/drug).",
    )

    design.add_argument(
        "--no-standardize-regressors",
        metavar="COLUMN",
        nargs='+',
        dest="no_standardize_regressors",
        help="Regressor column names to keep in original units (skip z-scoring). "
             "Useful for interpretability when coefficients need original scale (e.g., age IQ).",
    )

    design.add_argument(
        "--no-intercept",
        action="store_true",
        dest="no_intercept",
        help="Exclude the intercept from the design matrix. By default, an intercept is included.",
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
        metavar="ENTITY",
        dest="pair_by",
        default=None,
        help="BIDS entity key for pairing observations (default: 'sub' for subject). "
             "Supports BIDS abbreviations ('sub', 'ses', 'run', etc.) or full names. "
             "Examples: --pair-by sub, --pair-by ses, --pair-by run. "
             "Data filenames must contain matching entity values (e.g., '*ses-pre*').",
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

    # Note: No warning needed - pattern matching is the primary method now

    # Build configuration from CLI options
    config_overrides = {}

    if args.analysis_type:
        config_overrides["analysis_type"] = args.analysis_type
    if args.contrasts:
        config_overrides["contrasts"] = list(args.contrasts)

    # Participant filter
    if args.participant_label:
        config_overrides["participant_label"] = list(args.participant_label)
    
    # Design matrix configuration (for GLM)
    # Default: use all columns from participants.tsv if --regressors not specified
    # If --regressors specified: use only those columns
    # --categorical-regressors can specify columns not in --regressors (will be add them)
    design_matrix_config: Dict[str, Any] = {
        "add_intercept": not args.no_intercept,
        "standardize_continuous": True,
    }
    
    if args.regressors:
        # User specified columns - use these plus any additional categorical regressors
        columns = list(args.regressors)
        # Add categorical regressors not already in the list
        if args.categorical_regressors:
            for cat_col in args.categorical_regressors:
                if cat_col not in columns:
                    columns.append(cat_col)
        design_matrix_config["columns"] = columns
    else:
        # No --regressors specified: use "all" as a flag to use all participants.tsv columns
        design_matrix_config["columns"] = "all"
    
    if args.categorical_regressors:
        design_matrix_config["categorical_columns"] = list(args.categorical_regressors)
    if args.no_standardize_regressors:
        design_matrix_config["no_standardize_columns"] = list(args.no_standardize_regressors)
    
    config_overrides["design_matrix"] = design_matrix_config
    
    if args.verbose > 0:
        print(f"Design matrix configuration:")
        columns_cfg = design_matrix_config.get("columns")
        if columns_cfg == "all":
            print(f"  Regressors: ALL columns from participants.tsv")
        elif isinstance(columns_cfg, list):
            print(f"  Regressors: {', '.join(columns_cfg)}")
        if args.categorical_regressors:
            print(f"  Categorical regressors (dummy-coded): {', '.join(args.categorical_regressors)}")
        if args.no_standardize_regressors:
            print(f"  No z-scoring applied to: {', '.join(args.no_standardize_regressors)}")
        if args.no_intercept:
            print(f"  Intercept: excluded (--no-intercept)")
        else:
            print(f"  Intercept: included")
    
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
            participants_file=str(args.participants_file) if args.participants_file else None,
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
