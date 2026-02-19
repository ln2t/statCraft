"""
Main pipeline module for StatCraft.

This module provides a high-level interface for running
complete second-level neuroimaging analyses.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from statcraft.config import Config, load_config
from statcraft.core.data_loader import DataLoader
from statcraft.core.design_matrix import DesignMatrixBuilder
from statcraft.core.glm import SecondLevelGLM, NormalizationMixin, ConnectivityGLM
from statcraft.core.inference import StatisticalInference, ConnectivityInference
from statcraft.core.annotation import ClusterAnnotator
from statcraft.core.report import ReportGenerator

logger = logging.getLogger(__name__)


def _print_file_list_limited(items, prefix="  ", max_head=5, max_tail=4):
    """
    Print a list of items with ellipsis if the list is too long.

    Shows first `max_head` and last `max_tail` items, with a summary
    of how many items were hidden in between.

    Parameters
    ----------
    items : list
        List of items to print (usually filenames or file info dicts)
    prefix : str
        Prefix to add before each line (default: "  ")
    max_head : int
        Number of items to show at the start (default: 5)
    max_tail : int
        Number of items to show at the end (default: 4)
    """
    total = len(items)

    if total <= max_head + max_tail:
        # Show all items
        for item in items:
            print(f"{prefix}{item}")
    else:
        # Show first max_head items
        for item in items[:max_head]:
            print(f"{prefix}{item}")

        # Show how many are hidden
        hidden = total - max_head - max_tail
        print(f"{prefix}({hidden} more files)")

        # Show last max_tail items
        for item in items[-max_tail:]:
            print(f"{prefix}{item}")


class StatCraftPipeline:
    """
    High-level pipeline for second-level neuroimaging analysis.
    
    This class orchestrates the complete analysis workflow:
    1. Data loading and validation
    2. Design matrix construction
    3. GLM fitting / t-tests
    4. Statistical inference
    5. Cluster annotation
    6. Report generation
    
    Parameters
    ----------
    bids_dir : str or Path
        Path to BIDS rawdata directory.
    output_dir : str or Path
        Path to output directory for StatCraft results.
    derivatives : list of str or Path
        Paths to derivative folders containing images.
    config : Config, str, Path, or dict, optional
        Configuration (Config object, path to config file, or dict).
    participants_file : str or Path, optional
        Path to a custom participants.tsv file. If not provided, will look for
        participants.tsv in bids_dir. Useful when bids_dir is directly a
        derivatives folder without participants.tsv.
    
    Attributes
    ----------
    config : Config
        Configuration object.
    data_loader : DataLoader
        Data loader instance.
    glm : SecondLevelGLM
        GLM model instance.
    inference : StatisticalInference
        Statistical inference instance.
    annotator : ClusterAnnotator
        Cluster annotator instance.
    report : ReportGenerator
        Report generator instance.
    """
    
    def __init__(
        self,
        bids_dir: Union[str, Path],
        output_dir: Union[str, Path],
        derivatives: List[Union[str, Path]],
        config: Optional[Union[Config, str, Path, Dict]] = None,
        participants_file: Optional[Union[str, Path]] = None,
    ):
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.derivatives = [Path(d) for d in derivatives]
        self.participants_file = Path(participants_file) if participants_file else None
        
        # Load configuration
        if config is None:
            self.config = Config()
        elif isinstance(config, Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = Config(**config)
        else:
            self.config = load_config(config)
        
        # Set up logging
        self._setup_logging()
        
        logger.info(f"Initializing StatCraft pipeline")
        logger.info(f"BIDS directory: {self.bids_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Derivatives: {self.derivatives}")
        if self.participants_file:
            logger.info(f"Participants file: {self.participants_file}")
        
        # Initialize components
        self.data_loader = DataLoader(
            bids_dir=self.bids_dir,
            derivatives=self.derivatives,
            output_dir=self.output_dir,
            participants_file=self.participants_file,
            analysis_type=self.config.get("analysis_type", "glm"),
        )
        
        self.glm: Optional[SecondLevelGLM] = None
        self.connectivity_glm: Optional[ConnectivityGLM] = None
        self.design_matrix_builder: Optional[DesignMatrixBuilder] = None
        self.inference: Optional[StatisticalInference] = None
        self.connectivity_inference: Optional[ConnectivityInference] = None
        self.annotator: Optional[ClusterAnnotator] = None
        self.report: Optional[ReportGenerator] = None
        
        # Storage for results
        self._all_images: List[Dict] = []  # All discovered images
        self._images: List[Dict] = []  # Valid images after validation
        self._invalid_images: List[Dict] = []  # Invalid images
        self._image_paths: List[str] = []
        self._design_matrix: Optional[pd.DataFrame] = None
        self._results: Dict[str, Any] = {}
        self._scaling_key: Optional[str] = None  # Scaling key for output filenames
        self._glm_input_images: Optional[List] = None  # Images fed to the GLM (for saving to data/)
        self._data_type: str = "nifti"  # 'nifti' or 'connectivity'
        self._connectivity_metadata: Optional[Dict[str, Any]] = None  # ROI coords, names, atlas
    
    def _setup_logging(self) -> None:
        """Configure logging based on verbosity setting."""
        verbose = self.config.get("verbose", 1)

        if verbose == 0:
            level = logging.WARNING
        elif verbose == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Always suppress matplotlib and fsspec debug/info spam regardless of verbosity
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('fsspec').setLevel(logging.WARNING)
    
    def load_data(
        self,
        participant_label: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
    ) -> List[Dict]:
        """
        Load and validate image data.

        Parameters
        ----------
        participant_label : list of str, optional
            List of participant labels to include. Uses config if not provided.
        pattern : str, optional
            Glob pattern for finding files.
        exclude_pattern : str, optional
            Glob pattern to exclude files from the match.

        Returns
        -------
        list of dict
            List of valid image info dictionaries.
        """
        logger.info("Loading data...")

        # Use config participant_label if not provided
        if participant_label is None:
            participant_label = self.config.get("participant_label")

        if pattern is None:
            pattern = self.config.get("file_pattern")

        if exclude_pattern is None:
            exclude_pattern = self.config.get("exclude_pattern")

        # Get images
        images = self.data_loader.get_images(
            participant_label=participant_label,
            pattern=pattern,
            exclude_pattern=exclude_pattern,
        )

        if not images:
            raise ValueError("No images found matching the specified criteria")

        # Log discovered files summary (detailed output via print statement below)
        logger.info(f"Discovered {len(images)} image files matching criteria")

        # Output to console/stdout for immediate visibility with limited file list
        print(f"\nDiscovered {len(images)} image files matching criteria:")
        print("=" * 80)
        # Prepare display list with limited output for large file counts
        file_list = []
        for i, img in enumerate(images, 1):
            # Show relevant BIDS entities if available
            entities_str = ""
            if 'entities' in img:
                entities = img['entities']
                parts = []
                if 'subject' in entities:
                    parts.append(f"sub-{entities['subject']}")
                if 'session' in entities:
                    parts.append(f"ses-{entities['session']}")
                if 'task' in entities:
                    parts.append(f"task-{entities['task']}")
                if parts:
                    entities_str = f" ({', '.join(parts)})"
            file_list.append(f"[{i:3d}] {img['path']}{entities_str}")
        _print_file_list_limited(file_list, prefix="  ")
        print("=" * 80)
        print()

        # Check if this is connectivity data
        first_data_type = images[0].get('data_type', 'nifti')
        is_connectivity = first_data_type == 'connectivity'

        # Validate based on data type
        if is_connectivity:
            valid_images, invalid_images = self.data_loader.validate_connectivity_matrices(images)
            validation_name = "connectivity matrix"
        else:
            valid_images, invalid_images = self.data_loader.validate_mni_space(images)
            validation_name = "MNI space"

        if not valid_images:
            raise ValueError(f"No valid images after {validation_name} validation")

        if invalid_images:
            logger.warning(f"Excluded {len(invalid_images)} invalid images")
            print(f"\nValidation: {len(invalid_images)} image(s) excluded:")
            print("-" * 80)
            for i, img in enumerate(invalid_images, 1):
                print(f"  [X] {img['path']}")
                print(f"      Reason: {img.get('reason', 'Unknown')}")
            print("-" * 80)
            print()

        # Store both all images and valid images for reporting
        self._all_images = images
        self._images = valid_images
        self._image_paths = [str(img["path"]) for img in valid_images]
        self._invalid_images = invalid_images

        # Detect data type from loaded images
        self._detect_data_type()

        # Clear image cache after validation to free memory
        self.data_loader.clear_cache()

        print(f"✓ Validation complete: {len(self._images)} valid image(s) ready for analysis\n")
        logger.info(f"Loaded {len(self._images)} valid images")
        return self._images

    def _detect_data_type(self) -> str:
        """
        Detect the data type from loaded images.

        Updates self._data_type based on file extensions.
        If connectivity data, also loads ROI metadata.

        Returns
        -------
        str
            'nifti' or 'connectivity'
        """
        if not self._images:
            return "nifti"

        # Check first image to determine type
        first_image = self._images[0]
        data_type = first_image.get('data_type', 'nifti')

        # Verify all images have same type
        all_same_type = all(
            img.get('data_type', 'nifti') == data_type
            for img in self._images
        )

        if not all_same_type:
            raise ValueError(
                "Mixed data types detected. All files must be either NIfTI or connectivity matrices."
            )

        self._data_type = data_type
        logger.info(f"Detected data type: {self._data_type}")

        if self._data_type == 'connectivity':
            # Load connectivity metadata from first image
            self._connectivity_metadata = self.data_loader.get_connectivity_metadata(first_image)

            if self._connectivity_metadata.get('roi_coordinates') is not None:
                n_rois = len(self._connectivity_metadata['roi_coordinates'])
                logger.info(f"Loaded {n_rois} ROI coordinates from JSON sidecar")
            else:
                logger.warning("No ROI coordinates found in JSON sidecar - connectome plot will be unavailable")

            if self._connectivity_metadata.get('roi_names') is not None:
                logger.info(f"Loaded {len(self._connectivity_metadata['roi_names'])} ROI names")

            if self._connectivity_metadata.get('atlas_name') is not None:
                logger.info(f"Atlas: {self._connectivity_metadata['atlas_name']}")

        return self._data_type

    def load_connectivity_data(
        self,
        participant_label: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
    ) -> List[Dict]:
        """
        Load and validate connectivity matrix data (.npy files).

        Parameters
        ----------
        participant_label : list of str, optional
            List of participant labels to include.
        pattern : str, optional
            Glob pattern for finding files.
        exclude_pattern : str, optional
            Glob pattern to exclude files.

        Returns
        -------
        list of dict
            List of valid connectivity matrix info dictionaries.
        """
        logger.info("Loading connectivity data...")

        # Use config participant_label if not provided
        if participant_label is None:
            participant_label = self.config.get("participant_label")

        if pattern is None:
            pattern = self.config.get("file_pattern", "**/*.npy")

        if exclude_pattern is None:
            exclude_pattern = self.config.get("exclude_pattern")

        # Get .npy files
        images = self.data_loader.get_images(
            participant_label=participant_label,
            pattern=pattern,
            exclude_pattern=exclude_pattern,
            extension=".npy",
        )

        if not images:
            raise ValueError("No connectivity matrices found matching the specified criteria")

        # Log discovered files summary (detailed output via print statement below)
        logger.info(f"Discovered {len(images)} connectivity matrix files matching criteria")

        print(f"\nDiscovered {len(images)} connectivity matrix files:")
        print("=" * 80)
        # Prepare display list with limited output for large file counts
        file_list = [f"[{idx}/{len(images)}] {img['path']}"
                     for idx, img in enumerate(images, 1)]
        _print_file_list_limited(file_list, prefix="  ")
        print("=" * 80)
        print()

        # Validate connectivity matrices
        valid_images, invalid_images = self.data_loader.validate_connectivity_matrices(images)

        if not valid_images:
            raise ValueError("No valid connectivity matrices after validation")

        if invalid_images:
            logger.warning(f"Excluded {len(invalid_images)} invalid matrices")
            print(f"\nValidation: {len(invalid_images)} matrix(es) excluded:")
            for img in invalid_images:
                print(f"  [X] {img['path']}")
                print(f"      Reason: {img.get('reason', 'Unknown')}")
            print()

        # Store results
        self._all_images = images
        self._images = valid_images
        self._image_paths = [str(img["path"]) for img in valid_images]
        self._invalid_images = invalid_images
        self._data_type = "connectivity"

        # Load connectivity metadata
        self._connectivity_metadata = self.data_loader.get_connectivity_metadata(valid_images[0])

        # Clear cache
        self.data_loader.clear_cache()

        print(f"✓ Validation complete: {len(self._images)} valid connectivity matrix(es) ready for analysis\n")
        logger.info(f"Loaded {len(self._images)} valid connectivity matrices")
        return self._images
    
    def build_design_matrix(
        self,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build the design matrix for analysis.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns from participants.tsv to include.
        
        Returns
        -------
        pd.DataFrame
            Design matrix.
        """
        logger.info("Building design matrix...")
        
        analysis_type = self.config.get("analysis_type", "glm")
        
        # Get participant data only if it's required for this analysis type
        requires_participants = analysis_type in ["glm", "two-sample", "two_sample"]
        participants = None
        
        if requires_participants:
            participants = self.data_loader.get_participants_for_images(self._images)
        
        self.design_matrix_builder = DesignMatrixBuilder(participants)
        
        if analysis_type in ["one-sample", "one_sample"]:
            self._design_matrix = self.design_matrix_builder.build_one_sample_design_matrix(
                n_subjects=len(self._images)
            )
            
        elif analysis_type in ["two-sample", "two_sample"]:
            group_config = self.config.get("group_comparison", {})
            group_column = group_config.get("group_column")
            
            if group_column is None:
                raise ValueError("group_comparison.group_column required for two-sample analysis")
            
            group_labels = participants[group_column].tolist()
            group1 = group_config.get("group1")
            group2 = group_config.get("group2")
            
            if group1 and group2:
                group_names = (group1, group2)
            else:
                unique = list(set(group_labels))
                group_names = (unique[0], unique[1]) if len(unique) >= 2 else ("group1", "group2")
            
            self._design_matrix = self.design_matrix_builder.build_two_sample_design_matrix(
                group_labels=group_labels,
                group_names=group_names,
            )
            
        elif analysis_type == "paired":
            paired_config = self.config.get("paired_test", {})
            n_pairs = len(self._images) // 2
            self._design_matrix = self.design_matrix_builder.build_paired_design_matrix(
                n_pairs=n_pairs
            )
            
        else:  # GLM
            dm_config = self.config.get("design_matrix", {})
            columns = columns or dm_config.get("columns", [])
            
            # Handle "all" keyword: use all columns from participants.tsv
            if columns == "all":
                # Get all columns except 'participant_id' (and similar standard columns)
                excluded_cols = {'participant_id', 'subject_id', 'sub', 'subject'}
                columns = [c for c in participants.columns if c.lower() not in excluded_cols]
                logger.info(f"Using all columns from participants.tsv: {columns}")
            
            if not columns:
                # Default: one-sample (intercept only)
                self._design_matrix = self.design_matrix_builder.build_one_sample_design_matrix(
                    n_subjects=len(self._images)
                )
            else:
                self._design_matrix = self.design_matrix_builder.build_design_matrix(
                    columns=columns,
                    add_intercept=dm_config.get("add_intercept", True),
                    categorical_columns=dm_config.get("categorical_columns"),
                    standardize_continuous=dm_config.get("standardize_continuous", True),
                    no_standardize_columns=dm_config.get("no_standardize_columns"),
                )
        
        logger.info(f"Design matrix shape: {self._design_matrix.shape}")
        return self._design_matrix
    
    def add_contrasts(
        self,
        contrast_specs: Optional[List] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Add contrasts for hypothesis testing.
        
        Parameters
        ----------
        contrast_specs : list, optional
            Contrast specifications. Uses config if not provided.
        
        Returns
        -------
        dict
            Dictionary of contrast names to vectors.
        """
        if self.design_matrix_builder is None:
            raise ValueError("Must build design matrix before adding contrasts")
        
        if contrast_specs is None:
            contrast_specs = self.config.get("contrasts", [])
        
        # Also check singular "contrast" from CLI (--contrast option)
        if not contrast_specs:
            single_contrast = self.config.get("contrast")
            if single_contrast:
                contrast_specs = [single_contrast]
        
        if not contrast_specs:
            # Default contrast: test intercept / mean
            # Check if intercept exists in the design matrix
            if "intercept" not in self._design_matrix.columns:
                raise ValueError(
                    "Cannot use default 'mean' contrast when --no-intercept is specified. "
                    "Please provide explicit contrast(s) using --contrasts (e.g., --contrasts age sex_M-sex_F). "
                    "The default contrast tests the intercept, which is not present when --no-intercept is used."
                )
            logger.info("No contrasts specified, using default (intercept)")
            n_cols = len(self._design_matrix.columns)
            self.design_matrix_builder.contrasts["effectOfMean"] = np.array(
                [1.0] + [0.0] * (n_cols - 1)
            )
        else:
            self.design_matrix_builder.add_contrasts_from_config(contrast_specs)
        
        logger.info(f"Added {len(self.design_matrix_builder.contrasts)} contrasts")
        return self.design_matrix_builder.contrasts

    def _compute_intersection_mask(self) -> Optional[nib.Nifti1Image]:
        """
        Compute intersection mask from all masks used in z-scoring.

        This method takes all unique masks that were used during z-scoring normalization
        and computes their intersection using nilearn's intersect_masks function.
        The intersection mask contains only voxels that are valid (non-zero) in ALL
        individual masks.

        Returns
        -------
        nibabel.Nifti1Image or None
            Intersection mask image if masks are available, None otherwise.
        """
        # Check if masks were saved during z-scoring
        if not hasattr(self, '_zscore_masks_used') or not self._zscore_masks_used:
            logger.info("No z-scoring masks found, will use automatic mask computation")
            return None

        # Get unique masks
        unique_masks = list(set(self._zscore_masks_used.values()))

        if len(unique_masks) == 0:
            logger.info("No unique masks found, will use automatic mask computation")
            return None

        logger.info(f"Computing intersection mask from {len(unique_masks)} unique mask(s)")
        print(f"\nComputing intersection mask from {len(unique_masks)} unique mask file(s)...")

        # Load all masks
        mask_images = []
        for mask_path in unique_masks:
            try:
                mask_img = nib.load(mask_path)
                mask_images.append(mask_img)
                logger.debug(f"  Loaded mask: {Path(mask_path).name}")
            except Exception as e:
                logger.warning(f"Could not load mask {mask_path}: {e}")
                continue

        if len(mask_images) < 1:
            logger.warning("Could not load any masks for intersection, will use automatic mask computation")
            return None

        # Compute intersection
        try:
            from nilearn.masking import intersect_masks

            if len(mask_images) == 1:
                intersection = mask_images[0]
                logger.info("Only one mask found, using it directly as GLM mask")
            else:
                # intersect_masks takes a list of mask images and returns their intersection
                intersection = intersect_masks(mask_images, threshold=0)
                logger.info(f"Computed intersection of {len(mask_images)} masks")
                print(f"  Intersection mask computed successfully")
        except Exception as e:
            logger.error(f"Error computing mask intersection: {e}")
            logger.info("Will use automatic mask computation")
            return None

        return intersection

    def fit_model(
        self,
        mask: Optional[Union[str, Path, nib.Nifti1Image]] = None,
        smoothing_fwhm: Optional[float] = None,
    ) -> SecondLevelGLM:
        """
        Fit the second-level GLM model.
        
        Parameters
        ----------
        mask : str, Path, or nibabel image, optional
            Brain mask.
        smoothing_fwhm : float, optional
            Smoothing kernel FWHM.
        
        Returns
        -------
        SecondLevelGLM
            Fitted model.
        """
        logger.info("Fitting GLM model...")

        analysis_type = self.config.get("analysis_type", "glm")

        # Compute intersection mask from z-scoring masks if no explicit mask provided
        if mask is None:
            intersection_mask = self._compute_intersection_mask()
            if intersection_mask is not None:
                mask = intersection_mask
                logger.info("Using intersection mask from z-scoring masks for GLM")

        # Apply normalization if specified
        norm_config = self.config.get("normalization", {})
        images_to_use = self._image_paths

        if norm_config.get("method") == "mean":
            logger.info("Applying mean normalization...")
            norm_mask = norm_config.get("mask")
            normalized = NormalizationMixin.mean_normalize(
                self._image_paths,
                mask=norm_mask,
            )
            images_to_use = normalized

        # Create GLM
        # Use smoothing_fwhm from parameter, or fall back to config value
        if smoothing_fwhm is None:
            smoothing_fwhm = self.config.get("glm", {}).get("smoothing_fwhm", 0)

        self.glm = SecondLevelGLM(
            mask=mask,
            smoothing_fwhm=smoothing_fwhm,
        )
        
        # Handle paired analysis specially
        if analysis_type == "paired":
            paired_config = self.config.get("paired_test", {})
            pairs = self.data_loader.organize_paired_data(
                self._images,
                pair_by=paired_config["pair_by"],
                condition1=paired_config["condition1"],
                condition2=paired_config["condition2"],
            )

            images_cond1 = [str(p[0]["path"]) for p in pairs]
            images_cond2 = [str(p[1]["path"]) for p in pairs]

            # Store for permutation testing
            self._paired_images_cond1 = images_cond1
            self._paired_images_cond2 = images_cond2

            # Store input images for saving to data/ folder
            self._glm_input_images = images_cond1 + images_cond2

            contrast_name = f"{paired_config['condition1']}_vs_{paired_config['condition2']}"
            self.glm.paired_ttest(
                images_cond1,
                images_cond2,
                condition1_name=paired_config["condition1"],
                condition2_name=paired_config["condition2"],
                contrast_name=contrast_name,
            )
        else:
            # Standard GLM fitting
            self.glm.fit(images_to_use, self._design_matrix)

            # Store input images for saving to data/ folder
            self._glm_input_images = images_to_use

            # Compute contrasts
            for name, vector in self.design_matrix_builder.contrasts.items():
                self.glm.compute_contrast(vector, contrast_name=name)

        logger.info("Model fitted successfully")
        return self.glm
    
    def run_inference(
        self,
        include_permutation: bool = None,
    ) -> StatisticalInference:
        """
        Run statistical inference on model results.
        
        Parameters
        ----------
        include_permutation : bool, optional
            Whether to run permutation testing.
        
        Returns
        -------
        StatisticalInference
            Inference results.
        """
        logger.info("Running statistical inference...")
        
        if self.glm is None:
            raise ValueError("Must fit model before running inference")
        
        inf_config = self.config.get("inference", {})
        
        self.inference = StatisticalInference(
            alpha_corrected=inf_config.get("alpha_corrected", 0.05),
            alpha_uncorrected=inf_config.get("alpha_uncorrected", 0.001),
            cluster_threshold=inf_config.get("cluster_threshold", 10),
            two_sided=inf_config.get("two_sided", True),
        )
        
        if include_permutation is None:
            include_permutation = inf_config.get("run_permutation", False)
        
        corrections = inf_config.get("corrections", ["uncorrected", "fdr", "bonferroni"])
        
        # Run inference for each contrast
        for contrast_name, result in self.glm.results.items():
            # Use z_score for thresholding (required by nilearn.glm.threshold_stats_img)
            # Fall back to stat if z_score is not available
            if "z_score" in result["maps"]:
                z_map = result["maps"]["z_score"]
            else:
                logger.warning(f"z_score not available for {contrast_name}, using stat map (not recommended)")
                z_map = result["maps"]["stat"]

            # Standard corrections
            if "uncorrected" in corrections:
                self.inference.threshold_uncorrected(z_map, contrast_name=contrast_name)

            if "fdr" in corrections:
                self.inference.threshold_fdr(z_map, contrast_name=contrast_name)

            if "bonferroni" in corrections:
                self.inference.threshold_fwer_bonferroni(z_map, contrast_name=contrast_name)

            # Extra cluster analysis (if enabled)
            if self.config.get("extra_cluster_analysis", False):
                logger.info(f"Running extra cluster analysis for {contrast_name}...")
                print(f"\nPerforming detailed cluster analysis for {contrast_name}...")

                overlap_threshold = self.config.get("cluster_overlap_threshold", 5.0)

                # Initialize storage for enhanced cluster tables
                if not hasattr(self.inference, 'enhanced_cluster_tables'):
                    self.inference.enhanced_cluster_tables = {}
                if not hasattr(self.inference, 'cortical_summaries'):
                    self.inference.cortical_summaries = {}

                # Run extra analysis for each correction method that produced clusters
                for correction in corrections:
                    if correction in self.inference.thresholded_maps.get(contrast_name, {}):
                        thresholded_map = self.inference.thresholded_maps[contrast_name][correction]
                        cluster_table = self.inference.cluster_tables.get(contrast_name, {}).get(correction)

                        if cluster_table is not None and not cluster_table.empty:
                            try:
                                enhanced_table, cortical_summary = self.inference.perform_extra_cluster_analysis(
                                    stat_map=z_map,
                                    thresholded_map=thresholded_map,
                                    contrast_name=contrast_name,
                                    correction=correction,
                                    overlap_threshold=overlap_threshold,
                                )

                                # Store results
                                if contrast_name not in self.inference.enhanced_cluster_tables:
                                    self.inference.enhanced_cluster_tables[contrast_name] = {}
                                if contrast_name not in self.inference.cortical_summaries:
                                    self.inference.cortical_summaries[contrast_name] = {}

                                self.inference.enhanced_cluster_tables[contrast_name][correction] = enhanced_table
                                self.inference.cortical_summaries[contrast_name][correction] = cortical_summary

                                print(f"  ✓ {correction}: Analyzed {len(enhanced_table)} clusters")
                            except Exception as e:
                                logger.warning(f"Extra cluster analysis failed for {contrast_name}/{correction}: {e}")
                                print(f"  ✗ {correction}: Analysis failed - {e}")

                print(f"✓ Extra cluster analysis completed")

            # Permutation testing
            if include_permutation:
                logger.info(f"Running permutation test for {contrast_name}...")
                print(f"\nRunning permutation test with {inf_config.get('n_permutations', 1000)} permutations...")

                # For paired tests, use difference images (one-sample test on differences)
                analysis_type = self.config.get("analysis_type", "glm")

                # Check for pattern-based paired test (has pre-computed difference images)
                if hasattr(self, '_diff_images'):
                    logger.info("Detected pattern-based paired test - using pre-computed difference images")
                    print("  Using paired-sample permutation test (based on difference images)")
                    perm_input = self._diff_images
                    perm_design = self._design_matrix  # Already has correct size (n_subjects x 1)
                    perm_contrast = [1.0]  # Test if mean difference != 0

                # Check for legacy paired test (condition1/condition2)
                elif analysis_type == "paired" and hasattr(self, '_paired_images_cond1'):
                    logger.info("Detected legacy paired test - computing difference images for permutation test")
                    print("  Using paired-sample permutation test (based on difference images)")

                    # Compute difference images
                    from nilearn.image import math_img
                    import numpy as np

                    diff_images = []
                    for img1, img2 in zip(self._paired_images_cond1, self._paired_images_cond2):
                        diff_img = math_img("img1 - img2", img1=img1, img2=img2)
                        diff_images.append(diff_img)

                    # Create simple design matrix for one-sample test (just intercept)
                    import pandas as pd
                    n_samples = len(diff_images)
                    diff_design_matrix = pd.DataFrame({'intercept': np.ones(n_samples)})
                    diff_contrast = [1.0]  # Test if mean difference != 0

                    perm_input = diff_images
                    perm_design = diff_design_matrix
                    perm_contrast = diff_contrast

                else:
                    # Standard unpaired permutation test
                    perm_input = self._image_paths
                    perm_design = self._design_matrix
                    perm_contrast = self.design_matrix_builder.contrasts.get(contrast_name, [1.0])

                try:
                    thresholded_map, cluster_table, additional_results = self.inference.threshold_fwer_permutation(
                        second_level_input=perm_input,
                        design_matrix=perm_design,
                        contrast=perm_contrast,
                        contrast_name=contrast_name,
                        n_perm=inf_config.get("n_permutations", 1000),
                        n_jobs=self.config.get("n_jobs", 1),
                        random_state=self.config.get("random_state"),
                        smoothing_fwhm=self.glm.smoothing_fwhm if self.glm else None,
                    )
                    print(f"✓ Permutation test completed successfully")
                    print(f"  Found {len(cluster_table)} significant clusters")
                    print(f"  Results stored with key 'fwer_perm'")
                except Exception as e:
                    logger.warning(f"Permutation test failed: {e}")
                    print(f"✗ Permutation test failed: {e}")
                    import traceback
                    print(f"  Traceback: {traceback.format_exc()}")
        
        logger.info("Statistical inference completed")
        return self.inference
    
    def annotate_clusters(
        self,
        atlas: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Annotate cluster tables with anatomical labels.
        
        Parameters
        ----------
        atlas : str, optional
            Atlas to use for annotation.
        
        Returns
        -------
        dict
            Dictionary of annotated cluster tables.
        """
        logger.info("Annotating clusters...")
        
        if self.inference is None:
            raise ValueError("Must run inference before annotating clusters")
        
        if atlas is None:
            atlas = self.config.get("atlas", "harvard_oxford")
        
        self.annotator = ClusterAnnotator(atlas=atlas)
        
        annotated_tables = {}
        
        for contrast_name, corrections in self.inference.cluster_tables.items():
            for correction, table in corrections.items():
                if len(table) > 0:
                    annotated = self.annotator.annotate_cluster_table(table)
                    self.inference.cluster_tables[contrast_name][correction] = annotated
                    annotated_tables[f"{contrast_name}_{correction}"] = annotated
        
        logger.info(f"Annotated {len(annotated_tables)} cluster tables")
        return annotated_tables
    
    def generate_report(
        self,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate HTML report.

        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses BIDS prefix with _report.html.

        Returns
        -------
        Path
            Path to generated report.
        """
        logger.info("Generating report...")

        # Generate BIDS-compatible filename
        if filename is None:
            bids_prefix = self._generate_bids_prefix()

            # Include contrast name(s) in filename
            contrast_names = list(self.glm.results.keys())
            if len(contrast_names) == 1:
                # Single contrast: use specific name
                contrast_str = contrast_names[0]
            elif len(contrast_names) > 1:
                # Multiple contrasts: join with +
                contrast_str = "+".join(contrast_names)
            else:
                # No contrasts (shouldn't happen, but fallback)
                contrast_str = "none"

            filename = f"{bids_prefix}_contrast-{contrast_str}_report.html"
        
        # Build report title with contrast information
        if self.glm and self.glm.results:
            contrast_names = list(self.glm.results.keys())
            if len(contrast_names) == 1:
                report_title = f"StatCraft Analysis Report — Contrast: {contrast_names[0]}"
            elif len(contrast_names) > 1:
                report_title = f"StatCraft Analysis Report — Contrasts: {', '.join(contrast_names)}"
            else:
                report_title = "StatCraft Second-Level Analysis Report"
        else:
            report_title = "StatCraft Second-Level Analysis Report"
        
        self.report = ReportGenerator(
            title=report_title,
            output_dir=self.output_dir,
        )

        # Collect masks and scaling ROIs used (pass the full dictionaries for mapping)
        masks_used_dict = None
        if hasattr(self, '_zscore_masks_used') and self._zscore_masks_used:
            masks_used_dict = self._zscore_masks_used

        scaling_rois_used_dict = None
        if hasattr(self, '_scaling_rois_used') and self._scaling_rois_used:
            scaling_rois_used_dict = self._scaling_rois_used

        # ===================================================================
        # REPORT STRUCTURE (Hierarchical organization)
        # ===================================================================
        
        # --------------------------------------------------------------------
        # ANALYSIS SECTION (Level 1)
        # --------------------------------------------------------------------
        self.report.add_section("Analysis", "", "text", level=1)
        
        # Add data files section (Level 2)
        self.report.add_data_files_section(
            all_images=self._all_images,
            valid_images=self._images,
            invalid_images=self._invalid_images,
            masks=masks_used_dict,
            scaling_rois=scaling_rois_used_dict,
        )

        # Add methodology section (Level 2)
        inf_config = self.config.get("inference", {})
        corrections = inf_config.get("corrections", ["uncorrected", "fdr", "bonferroni"])

        # Get preprocessing parameters
        glm_config = self.config.get("glm", {})
        smoothing_fwhm = glm_config.get("smoothing_fwhm", 0)
        zscore_applied = bool(self.config.get("zscore", False))
        scaling_applied = self.config.get("scaling_pattern", None)
        if scaling_applied and self.config.get("scaling_key"):
            scaling_applied = self.config.get("scaling_key")

        # Get paired information if available
        paired_info = None
        if hasattr(self, '_paired_info'):
            paired_info = self._paired_info

        self.report.add_methodology_section(
            design_matrix=self._design_matrix,
            contrasts=self.design_matrix_builder.contrasts,
            analysis_type=self.config.get("analysis_type", "glm"),
            n_subjects=len(self._images),
            correction_methods=corrections,
            alpha_corrected=inf_config.get("alpha_corrected", 0.05),
            alpha_uncorrected=inf_config.get("alpha_uncorrected", 0.001),
            smoothing_fwhm=smoothing_fwhm,
            zscore=zscore_applied,
            scaling=scaling_applied,
            n_contrasts=len(self.glm.results),
            paired_info=paired_info,
        )

        # --------------------------------------------------------------------
        # RESULTS SECTION (Level 1)
        # --------------------------------------------------------------------
        self.report.add_section("Results", "", "text", level=1)
        
        # Add results for each contrast
        for contrast_name, result in self.glm.results.items():
            stat_map = result["maps"]["stat"]
            thresholded_maps = self.inference.thresholded_maps.get(contrast_name, {})
            
            # Contrast section (Level 2)
            self.report.add_section(f"Contrast: {contrast_name}", "", "text", level=2)
            
            # Contrast Definition (Level 3)
            contrast_vector = self.design_matrix_builder.contrasts.get(contrast_name)
            if contrast_vector is not None:
                contrast_html = "<p><strong>Contrast weights:</strong></p><ul>"
                for i, val in enumerate(contrast_vector):
                    if val != 0:
                        col_name = self._design_matrix.columns[i] if i < len(self._design_matrix.columns) else f"Column {i}"
                        contrast_html += f"<li>{col_name}: {val:+.2f}</li>"
                contrast_html += "</ul>"
                self.report.add_section(f"Contrast Definition", contrast_html, "text", level=3)
            
            # Unthresholded maps (Level 3)
            if not self.report._is_map_empty(stat_map):
                unthresh_caption = (
                    "Unthresholded z-score map showing all voxels (positive and negative effects). "
                    "Color intensity represents effect size; no statistical threshold applied. "
                    "All data are displayed to show the full spatial extent of effects."
                )
                
                # Glass brain
                fig_key, vmin, vmax = self.report._plot_glass_brain(
                    stat_map,
                    title=f"{contrast_name} (Unthresholded) - Glass Brain",
                    threshold=0,
                )
                self.report.add_section(
                    "Unthresholded - Glass Brain",
                    fig_key,
                    "figure",
                    level=3,
                    caption=unthresh_caption,
                )

                # Lightbox
                fig_key = self.report._plot_axial_slices(
                    stat_map,
                    title=f"{contrast_name} (Unthresholded) - Axial Slices",
                    threshold=0,
                    vmin=vmin,
                    vmax=vmax,
                )
                self.report.add_section(
                    "Unthresholded - Lightbox",
                    fig_key,
                    "figure",
                    level=3,
                    caption=unthresh_caption,
                )
            
            # Thresholded maps - one section per correction method (Level 3)
            for correction, thresh_map in thresholded_maps.items():
                if not self.report._is_map_empty(thresh_map):
                    # Correction method header (Level 3)
                    correction_title = f"{correction.upper()} Corrected"
                    if correction == "uncorrected":
                        correction_title = f"Uncorrected (p < {inf_config.get('alpha_uncorrected', 0.001)})"
                    elif correction == "fwer_perm":
                        correction_title = f"Permutation-based FWER (p < {inf_config.get('alpha_corrected', 0.05)})"
                    
                    self.report.add_section(correction_title, "", "text", level=3)
                    
                    # Create informative caption based on correction method
                    if correction == "uncorrected":
                        thresh_caption = (
                            f"Thresholded z-score map showing voxels surviving uncorrected threshold (p < {inf_config.get('alpha_uncorrected', 0.001)}). "
                            "Only voxels with |z| > 1.96 are displayed. "
                            "No correction for multiple comparisons applied; results may include false positives."
                        )
                    elif correction == "fdr":
                        thresh_caption = (
                            f"Thresholded z-score map showing voxels surviving FDR correction (q < {inf_config.get('alpha_corrected', 0.05)}). "
                            "False Discovery Rate controls the expected proportion of false positives among suprathreshold voxels. "
                            "Only statistically significant voxels after correction are displayed."
                        )
                    elif correction == "bonferroni":
                        thresh_caption = (
                            f"Thresholded z-score map showing voxels surviving Bonferroni correction (p < {inf_config.get('alpha_corrected', 0.05)}, family-wise). "
                            "Controls family-wise error rate with very conservative threshold. "
                            "Only voxels passing stringent correction for multiple comparisons are shown."
                        )
                    elif correction == "fwer_perm":
                        thresh_caption = (
                            f"Thresholded z-score map showing voxels surviving permutation-based FWER correction (p < {inf_config.get('alpha_corrected', 0.05)}). "
                            "Non-parametric permutation testing controls family-wise error rate without parametric assumptions. "
                            "Only cluster-level significant results from permutation distribution are displayed."
                        )
                    elif correction == "perm":
                        thresh_caption = (
                            f"Thresholded z-score map showing voxels surviving permutation-based correction (p < {inf_config.get('alpha_corrected', 0.05)}, FWER). "
                            "Non-parametric permutation testing controls family-wise error rate. "
                            "Only cluster-level significant results are displayed."
                        )
                    else:
                        thresh_caption = (
                            f"Thresholded z-score map showing voxels surviving {correction.upper()} correction. "
                            "Only statistically significant voxels are displayed."
                        )
                    
                    # Glass brain (Level 4 - under correction method)
                    fig_key, vmin, vmax = self.report._plot_glass_brain(
                        thresh_map,
                        title=f"{contrast_name} ({correction.upper()}) - Glass Brain",
                        threshold=1.96,
                    )
                    self.report.add_section(
                        "Glass Brain",
                        fig_key,
                        "figure",
                        level=3,
                        caption=thresh_caption,
                    )

                    # Lightbox (Level 4)
                    fig_key = self.report._plot_axial_slices(
                        thresh_map,
                        title=f"{contrast_name} ({correction.upper()}) - Axial Slices",
                        threshold=1.96,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    self.report.add_section(
                        "Lightbox",
                        fig_key,
                        "figure",
                        level=3,
                        caption=thresh_caption,
                    )
                    
                    # Cluster table (Level 4)
                    cluster_table = self.inference.cluster_tables.get(contrast_name, {}).get(correction)
                    if cluster_table is not None:
                        if len(cluster_table) > 0 and 'Peak Stat' in cluster_table.columns:
                            n_positive = (cluster_table['Peak Stat'] > 0).sum()
                            n_negative = (cluster_table['Peak Stat'] < 0).sum()
                            logger.info(f"Adding to report - {contrast_name}/{correction}: {n_positive} positive, {n_negative} negative clusters")

                        self.report.add_cluster_table(
                            cluster_table=cluster_table,
                            contrast_name=contrast_name,
                            correction=correction,
                        )

                    # Enhanced cluster table (Level 4, if available)
                    if hasattr(self.inference, 'enhanced_cluster_tables'):
                        enhanced_table = self.inference.enhanced_cluster_tables.get(contrast_name, {}).get(correction)
                        if enhanced_table is not None:
                            self.report.add_enhanced_cluster_table(
                                enhanced_table=enhanced_table,
                                contrast_name=contrast_name,
                                correction=correction,
                            )

                    # Cortical/Non-Cortical summary (Level 4, if available)
                    if hasattr(self.inference, 'cortical_summaries'):
                        cortical_summary = self.inference.cortical_summaries.get(contrast_name, {}).get(correction)
                        if cortical_summary is not None:
                            self.report.add_cortical_summary(
                                cortical_summary=cortical_summary,
                                contrast_name=contrast_name,
                                correction=correction,
                            )

            # Add permutation null distribution visualization if available
            if hasattr(self.inference, '_inference_results'):
                perm_key = f"{contrast_name}_perm"
                if perm_key in self.inference._inference_results:
                    perm_results = self.inference._inference_results[perm_key]
                    logger.info(f"Checking permutation results for {contrast_name}: keys = {perm_results.keys()}")
                    if "h0_max_t" in perm_results and perm_results["h0_max_t"] is not None:
                        h0_dist = perm_results["h0_max_t"]
                        logger.info(f"Found h0_max_t with shape/length: {np.shape(h0_dist)}")
                        print(f"  Adding null distribution plot for {contrast_name} ({len(np.asarray(h0_dist).ravel())} values)")
                        self.report.add_permutation_null_distribution(
                            h0_distribution=h0_dist,
                            alpha=perm_results.get("alpha", 0.05),
                            contrast_name=contrast_name,
                            n_perm=perm_results.get("n_perm", 0),
                        )
                    else:
                        logger.warning(f"No h0_max_t found in permutation results for {contrast_name}")
                        print(f"  Warning: No null distribution available for {contrast_name}")
                else:
                    logger.info(f"No permutation results found for {contrast_name} (key: {perm_key})")

        # --------------------------------------------------------------------
        # CITATION SECTION (Level 1)
        # --------------------------------------------------------------------
        self.report.add_citation_section()

        # Add technical section with command line and config file link
        complete_config = self._create_reproducible_config()
        cli_command = complete_config.get('equivalent_cli_command', 'statcraft --config <config_file>')

        # Determine config filename
        bids_prefix = self._generate_bids_prefix()
        contrasts = self.design_matrix_builder.contrasts
        if contrasts:
            contrast_str = "_".join([c.replace(" ", "_") for c in contrasts.keys()])[:50]
        else:
            contrast_str = "none"
        config_filename = f"{bids_prefix}_contrast-{contrast_str}_config.yaml"

        # Try to read the config file content for embedding
        config_content = None
        config_path = self.output_dir / config_filename
        if config_path.exists():
            try:
                config_content = config_path.read_text()
                logger.info(f"Loaded config file for embedding: {config_path}")
            except Exception as e:
                logger.warning(f"Could not read config file for embedding: {e}")

        self.report.add_technical_section(
            command_line=cli_command,
            config_filename=config_filename,
            config_content=config_content,
        )

        # Save report
        report_path = self.report.save(filename)

        logger.info(f"Report saved to: {report_path}")
        return report_path
    
    def _generate_bids_prefix(self) -> str:
        """
        Generate BIDS-compatible prefix from input filenames.

        Extracts common parts from all input filenames and creates a prefix.
        Only includes BIDS entities that have the SAME value across all input files.
        This ensures that sample-specific entities (e.g., different 'label' values
        in multi-sample comparisons) are excluded from output filenames.

        For example, if comparing files with 'label-GS' and 'label-SSS', the
        'label' entity will be excluded from the output prefix since it has
        two different values.

        Returns
        -------
        str
            BIDS prefix for output files, or "group" if no common entities found.
        """
        if not self._images:
            return "group"

        # Get all filenames
        filenames = [Path(img["path"]).name for img in self._images]

        # Split filenames into BIDS entities
        from collections import defaultdict
        entities = defaultdict(set)

        for filename in filenames:
            # Remove extension
            name = filename.replace('.nii.gz', '').replace('.nii', '')

            # Parse BIDS entities (key-value pairs separated by _)
            parts = name.split('_')
            for part in parts:
                if '-' in part:
                    key, value = part.split('-', 1)
                    entities[key].add(value)

        # Build prefix from entities that have only one unique value (common across all files)
        # This excludes sample-specific entities in multi-sample analyses
        prefix_parts = []
        for key in ['task', 'space', 'desc', 'label', 'res', 'den']:
            if key in entities and len(entities[key]) == 1:
                value = list(entities[key])[0]
                prefix_parts.append(f"{key}-{value}")

        # Add scaling key if present
        if hasattr(self, '_scaling_key') and self._scaling_key:
            prefix_parts.append(f"scaling-{self._scaling_key}")

        if prefix_parts:
            return '_'.join(prefix_parts)
        else:
            return "group"

    def _generate_data_filename(
        self,
        img_info: Dict,
        idx: int,
        preproc_type: Optional[str] = None,
    ) -> str:
        """
        Generate BIDS-style filename for input data saved to the data/ folder.

        Parameters
        ----------
        img_info : dict
            Image info dictionary containing 'path' and 'entities'.
        idx : int
            Index of the image in the list.
        preproc_type : str, optional
            Type of preprocessing applied: 'zscore', 'scaled', 'diff', or None.

        Returns
        -------
        str
            BIDS-style filename with all relevant entities.
        """
        entities = img_info.get('entities', {})
        orig_path = Path(img_info['path'])
        orig_name = orig_path.name

        # Build filename from BIDS entities in standard order
        # Standard BIDS entity order: sub, ses, task, acq, ce, rec, dir, run, echo, part, space, res, den, label, desc
        entity_order = ['sub', 'ses', 'task', 'acq', 'ce', 'rec', 'dir', 'run', 'echo', 'part', 'space', 'res', 'den', 'label', 'desc']

        # Map full entity names back to BIDS abbreviations
        name_to_abbrev = {
            'subject': 'sub',
            'session': 'ses',
            'task': 'task',
            'run': 'run',
            'space': 'space',
            'description': 'desc',
            'acquisition': 'acq',
            'reconstruction': 'rec',
            'direction': 'dir',
            'resolution': 'res',
            'density': 'den',
        }

        # Build parts list
        parts = []

        # First, try to extract entities from the original filename to preserve order
        # This handles cases where entities might have different naming conventions
        orig_stem = orig_name.replace('.nii.gz', '').replace('.nii', '')
        orig_parts = orig_stem.split('_')

        # Collect all key-value pairs from original filename
        orig_entities = {}
        suffix_part = None
        for part in orig_parts:
            if '-' in part:
                key, value = part.split('-', 1)
                orig_entities[key] = value
            else:
                # Last part without hyphen is typically the suffix (e.g., 'bold', 'cvr')
                suffix_part = part

        # Build filename using standard entity order
        for key in entity_order:
            if key in orig_entities:
                parts.append(f"{key}-{orig_entities[key]}")
            else:
                # Check if we have this entity under a different name in the entities dict
                for full_name, abbrev in name_to_abbrev.items():
                    if abbrev == key and full_name in entities:
                        parts.append(f"{key}-{entities[full_name]}")
                        break

        # Add preprocessing indicator if specified
        if preproc_type:
            parts.append(f"preproc-{preproc_type}")

        # Add index for uniqueness (but not for diff images, which are already unique by pair)
        if preproc_type != "diff":
            parts.append(f"idx-{idx:04d}")

        # Add original suffix if found, otherwise use 'input'
        if suffix_part:
            parts.append(suffix_part)
        else:
            parts.append("input")

        filename = '_'.join(parts) + '.nii.gz'
        return filename

    def _generate_json_sidecar(
        self,
        nifti_path: Path,
        stat_type: str,
        contrast_name: str,
        threshold_type: Optional[str] = None,
        threshold_value: Optional[float] = None,
        actual_threshold: Optional[float] = None,
    ) -> Path:
        """
        Generate JSON sidecar for a NIfTI file.

        Parameters
        ----------
        nifti_path : Path
            Path to the NIfTI file.
        stat_type : str
            Type of statistic (e.g., "stat", "effect_size", "p_value").
        contrast_name : str
            Name of the contrast.
        threshold_type : str, optional
            Type of threshold applied (e.g., "uncorrected", "fdr", "bonferroni").
        threshold_value : float, optional
            Threshold value used (p-value or alpha level).
        actual_threshold : float, optional
            Actual z-score threshold value computed by threshold_stats_img.

        Returns
        -------
        Path
            Path to the JSON sidecar file.
        """
        json_path = nifti_path.with_suffix('').with_suffix('.json')

        metadata = {
            "Description": f"Second-level {stat_type} map for contrast: {contrast_name}",
            "ContrastName": contrast_name,
            "StatisticType": stat_type,
            "AnalysisType": self.config.get("analysis_type", "glm"),
            "NumberOfSubjects": len(self._images),
            "Software": "StatCraft",
        }

        if threshold_type:
            metadata["ThresholdType"] = threshold_type
            metadata["Thresholded"] = True
        else:
            metadata["Thresholded"] = False

        if threshold_value is not None:
            if threshold_type in ["uncorrected", "fdr"]:
                metadata["ThresholdPValue"] = threshold_value
            else:
                metadata["ThresholdValue"] = threshold_value

        # Add actual z-score threshold computed by nilearn
        if actual_threshold is not None:
            if np.isfinite(actual_threshold):
                metadata["ActualZThreshold"] = float(actual_threshold)
            else:
                metadata["ActualZThreshold"] = None
                metadata["Note"] = "No voxels survived thresholding"

        # Add inference settings
        inf_config = self.config.get("inference", {})
        metadata["AlphaCorrected"] = inf_config.get("alpha_corrected", 0.05)
        metadata["AlphaUncorrected"] = inf_config.get("alpha_uncorrected", 0.001)
        metadata["ClusterThreshold"] = inf_config.get("cluster_threshold", 10)

        import json
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Created JSON sidecar: {json_path}")
        return json_path

    def _generate_cli_command(self, config_dict: Dict[str, Any]) -> str:
        """
        Generate the equivalent CLI command from configuration.

        Parameters
        ----------
        config_dict : dict
            Complete configuration dictionary.

        Returns
        -------
        str
            CLI command string.
        """
        cmd_parts = ["statcraft"]

        # Add positional arguments
        if 'bids_dir' in config_dict:
            cmd_parts.append(f"{config_dict['bids_dir']}")
        if 'output_dir' in config_dict:
            cmd_parts.append(f"{config_dict['output_dir']}")

        # Add derivatives
        if 'derivatives' in config_dict and config_dict['derivatives']:
            for deriv in config_dict['derivatives']:
                cmd_parts.append(f"-d {deriv}")

        # Add analysis type
        if config_dict.get('analysis_type'):
            cmd_parts.append(f"-t {config_dict['analysis_type']}")

        # Add contrasts
        if config_dict.get('contrasts') and len(config_dict['contrasts']) > 0:
            # For single contrast, use it directly
            contrast_str = config_dict['contrasts'][0] if len(config_dict['contrasts']) == 1 else config_dict['contrasts'][0]
            cmd_parts.append(f"-C '{contrast_str}'")
        elif config_dict.get('contrast'):
            cmd_parts.append(f"-C '{config_dict['contrast']}'")

        # Add pattern options
        if config_dict.get('file_pattern'):
            cmd_parts.append(f"-p '{config_dict['file_pattern']}'")

        if config_dict.get('sample_patterns'):
            patterns_str = ' '.join([f"{k}={v}" for k, v in config_dict['sample_patterns'].items()])
            cmd_parts.append(f"-P '{patterns_str}'")

        if config_dict.get('exclude_pattern'):
            if isinstance(config_dict['exclude_pattern'], dict):
                # Label-specific excludes
                excludes_str = ' '.join([f"{k}={v}" for k, v in config_dict['exclude_pattern'].items()])
                cmd_parts.append(f"-e '{excludes_str}'")
            else:
                cmd_parts.append(f"-e '{config_dict['exclude_pattern']}'")

        # Add scaling
        if config_dict.get('scaling_pattern'):
            if config_dict.get('scaling_key'):
                cmd_parts.append(f"--scaling '{config_dict['scaling_key']}={config_dict['scaling_pattern']}'")
            else:
                cmd_parts.append(f"--scaling '{config_dict['scaling_pattern']}'")

        # Add z-scoring
        if config_dict.get('zscore'):
            cmd_parts.append("--zscore")

        # Add participant filter
        participant_label = config_dict.get('participant_label')
        if participant_label:
            labels = participant_label if isinstance(participant_label, list) else [participant_label]
            for label in labels:
                cmd_parts.append(f"--participant-label {label}")

        # Add group comparison options
        if config_dict.get('group_comparison', {}).get('group_column'):
            cmd_parts.append(f"--group-column {config_dict['group_comparison']['group_column']}")

        # Add paired test options
        paired_test = config_dict.get('paired_test', {})
        if paired_test.get('pair_by'):
            cmd_parts.append(f"--pair-by {paired_test['pair_by']}")
        if paired_test.get('condition1'):
            cmd_parts.append(f"--condition1 {paired_test['condition1']}")
        if paired_test.get('condition2'):
            cmd_parts.append(f"--condition2 {paired_test['condition2']}")

        # Add inference settings
        inference = config_dict.get('inference', {})
        if inference.get('alpha_corrected') is not None and inference['alpha_corrected'] != 0.05:
            cmd_parts.append(f"--alpha-corrected {inference['alpha_corrected']}")
        if inference.get('alpha_uncorrected') is not None and inference['alpha_uncorrected'] != 0.001:
            cmd_parts.append(f"--alpha-uncorrected {inference['alpha_uncorrected']}")
        if inference.get('cluster_threshold') is not None and inference['cluster_threshold'] != 10:
            cmd_parts.append(f"--cluster-threshold {inference['cluster_threshold']}")
        if inference.get('run_permutation'):
            cmd_parts.append("--permutation")
        if inference.get('n_permutations') is not None and inference['n_permutations'] != 5000:
            cmd_parts.append(f"--n-permutations {inference['n_permutations']}")

        # Add atlas
        if config_dict.get('atlas') and config_dict['atlas'] != 'harvard_oxford':
            cmd_parts.append(f"--atlas {config_dict['atlas']}")

        # Add computational settings
        if config_dict.get('n_jobs') is not None and config_dict['n_jobs'] != 1:
            cmd_parts.append(f"--n-jobs {config_dict['n_jobs']}")

        # Add extra cluster analysis settings
        if config_dict.get('extra_cluster_analysis'):
            cmd_parts.append("--extra-cluster-analysis")
        if config_dict.get('cluster_overlap_threshold') is not None and config_dict['cluster_overlap_threshold'] != 5.0:
            cmd_parts.append(f"--cluster-overlap-threshold {config_dict['cluster_overlap_threshold']}")

        # Verbosity is always added if needed
        if config_dict.get('verbose'):
            verbose_level = config_dict['verbose']
            if verbose_level > 0:
                cmd_parts.append('-' + 'v' * verbose_level)

        return ' '.join(cmd_parts)

    def _save_normalized_images(
        self,
        original_sample1: List[Dict],
        original_sample2: List[Dict],
        normalized_sample1: List[nib.Nifti1Image],
        normalized_sample2: List[nib.Nifti1Image],
        sample1_name: str,
        sample2_name: str,
        preproc_type: str = "scaled",
    ) -> None:
        """
        Save normalized (scaled or z-scored) images to disk.

        Parameters
        ----------
        original_sample1 : list of dict
            Original image info for sample 1.
        original_sample2 : list of dict
            Original image info for sample 2.
        normalized_sample1 : list of Nifti1Image
            Normalized images for sample 1.
        normalized_sample2 : list of Nifti1Image
            Normalized images for sample 2.
        sample1_name : str
            Name of sample 1 (for directory naming).
        sample2_name : str
            Name of sample 2 (for directory naming).
        preproc_type : str
            Type of preprocessing ("scaled" or "zscore").
        """
        # Check if supplementary data saving is enabled
        save_supp_data = self.config.get("output", {}).get("save_supplementary_data", False)
        if not save_supp_data:
            logger.debug(f"Skipping {preproc_type} image saving (supplementary data disabled)")
            return

        normalized_dir = self.output_dir / "supplementary_data" / preproc_type
        normalized_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(normalized_sample1) + len(normalized_sample2)} {preproc_type} images...")
        print(f"\nSaving {preproc_type} images to supplementary_data/{preproc_type}/")

        saved_count = 0

        # Save sample 1 normalized images
        for idx, (img_info, normalized_img) in enumerate(zip(original_sample1, normalized_sample1)):
            filename = self._generate_data_filename(img_info, idx, preproc_type=preproc_type)
            filepath = normalized_dir / filename
            try:
                nib.save(normalized_img, filepath)
                saved_count += 1
                logger.debug(f"  Saved {preproc_type} image: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save {preproc_type} image {filepath}: {e}")

        # Save sample 2 normalized images
        for idx, (img_info, normalized_img) in enumerate(zip(original_sample2, normalized_sample2)):
            filename = self._generate_data_filename(img_info, idx, preproc_type=preproc_type)
            filepath = normalized_dir / filename
            try:
                nib.save(normalized_img, filepath)
                saved_count += 1
                logger.debug(f"  Saved {preproc_type} image: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save {preproc_type} image {filepath}: {e}")

        logger.info(f"  Saved {saved_count} {preproc_type} images to {normalized_dir}")
        print(f"  Saved {saved_count} {preproc_type} images")

    def _save_normalized_images_two_sample(
        self,
        original_images: List[Dict],
        normalized_images: List[nib.Nifti1Image],
        preproc_type: str = "scaled",
    ) -> None:
        """
        Save normalized (scaled or z-scored) images for two-sample tests.

        Parameters
        ----------
        original_images : list of dict
            Original image info.
        normalized_images : list of Nifti1Image
            Normalized images.
        preproc_type : str
            Type of preprocessing ("scaled" or "zscore").
        """
        # Check if supplementary data saving is enabled
        save_supp_data = self.config.get("output", {}).get("save_supplementary_data", False)
        if not save_supp_data:
            logger.debug(f"Skipping {preproc_type} image saving (supplementary data disabled)")
            return

        normalized_dir = self.output_dir / "supplementary_data" / preproc_type
        normalized_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(normalized_images)} {preproc_type} images...")
        print(f"\nSaving {preproc_type} images to supplementary_data/{preproc_type}/")

        saved_count = 0
        for idx, (img_info, normalized_img) in enumerate(zip(original_images, normalized_images)):
            filename = self._generate_data_filename(img_info, idx, preproc_type=preproc_type)
            filepath = normalized_dir / filename
            try:
                nib.save(normalized_img, filepath)
                saved_count += 1
                logger.debug(f"  Saved {preproc_type} image: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save {preproc_type} image {filepath}: {e}")

        logger.info(f"  Saved {saved_count} {preproc_type} images to {normalized_dir}")
        print(f"  Saved {saved_count} {preproc_type} images")

    def _create_reproducible_config(self) -> Dict[str, Any]:
        """
        Create a complete configuration dict including all runtime parameters.

        This ensures the saved config file can be used to exactly reproduce the analysis.

        Returns
        -------
        dict
            Complete configuration dictionary.
        """
        # Start with the base config
        config_dict = dict(self.config.data)

        # Add bids_dir, output_dir, and derivatives paths for reproducibility
        config_dict['bids_dir'] = str(self.bids_dir)
        config_dict['output_dir'] = str(self.output_dir)
        config_dict['derivatives'] = [str(d) for d in self.derivatives]

        # Update with runtime parameters if they were provided
        if hasattr(self, '_runtime_pattern') and self._runtime_pattern is not None:
            config_dict['file_pattern'] = self._runtime_pattern

        if hasattr(self, '_runtime_exclude_pattern') and self._runtime_exclude_pattern is not None:
            # Handle both string and dict exclude patterns
            if isinstance(self._runtime_exclude_pattern, dict):
                # For dict exclude patterns, add a note and the dict
                config_dict['exclude_pattern'] = self._runtime_exclude_pattern
            else:
                config_dict['exclude_pattern'] = self._runtime_exclude_pattern

        if hasattr(self, '_runtime_sample_patterns') and self._runtime_sample_patterns is not None:
            config_dict['sample_patterns'] = self._runtime_sample_patterns

        if hasattr(self, '_runtime_scaling') and self._runtime_scaling is not None:
            config_dict['scaling_pattern'] = self._runtime_scaling

        if hasattr(self, '_scaling_key') and self._scaling_key is not None:
            config_dict['scaling_key'] = self._scaling_key

        if hasattr(self, '_runtime_zscore') and self._runtime_zscore:
            config_dict['zscore'] = self._runtime_zscore

        if hasattr(self, '_runtime_participant_label') and self._runtime_participant_label is not None:
            # Use runtime participant_label
            config_dict['participant_label'] = self._runtime_participant_label

        # Generate equivalent CLI command
        cli_cmd = self._generate_cli_command(config_dict)

        # Format the CLI command with backslashes for multiline readability
        # Split into lines if too long (more than 80 chars)
        if len(cli_cmd) > 80:
            formatted_lines = []
            current_line = []
            current_length = 0

            for part in cli_cmd.split():
                part_len = len(part) + 1  # +1 for space
                if current_length + part_len > 80 and current_line:
                    # End current line with backslash
                    formatted_lines.append(' '.join(current_line) + ' \\')
                    current_line = [part]
                    current_length = part_len
                else:
                    current_line.append(part)
                    current_length += part_len

            # Add the last line without backslash
            if current_line:
                formatted_lines.append(' '.join(current_line))

            config_dict['equivalent_cli_command'] = '\n  '.join(formatted_lines)
        else:
            config_dict['equivalent_cli_command'] = cli_cmd

        return config_dict

    def save_results(self) -> Dict[str, Path]:
        """
        Save all results to disk with BIDS-compatible naming.

        Output structure:
        - output_dir/
            - maps/           # Statistical maps (.nii.gz) and JSON sidecars
            - tables/         # Cluster tables (.tsv)
            - data/           # Input images fed to the GLM
            - logs/           # Config, design matrix, file lists
            - *_report.html   # HTML report (in root)

        Returns
        -------
        dict
            Dictionary of saved file paths.
        """
        logger.info("Saving results...")

        saved_files = {}

        # Generate BIDS prefix
        bids_prefix = self._generate_bids_prefix()
        logger.info(f"Using BIDS prefix: {bids_prefix}")

        # Create subdirectories
        maps_dir = self.output_dir / "maps"
        tables_dir = self.output_dir / "tables"
        logs_dir = self.output_dir / "logs" / bids_prefix

        maps_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Check if supplementary data saving is enabled
        save_supp_data = self.config.get("output", {}).get("save_supplementary_data", False)

        # Only create supplementary data directory if saving is enabled
        supp_data_dir = self.output_dir / "supplementary_data" if save_supp_data else None
        if supp_data_dir:
            supp_data_dir.mkdir(parents=True, exist_ok=True)

        # Save input images that were fed to the GLM
        if (save_supp_data and
            hasattr(self, '_glm_input_images') and
            self._glm_input_images is not None and
            len(self._glm_input_images) > 0):
            logger.info(f"Saving {len(self._glm_input_images)} input images to supplementary_data/")
            print(f"\nSaving {len(self._glm_input_images)} input images to supplementary_data/")

            # Check if these are difference images (paired analysis)
            is_diff_images = hasattr(self, '_diff_images') and self._diff_images is not None
            logger.info(f"Difference images: {is_diff_images}")

            # Create dedicated subdirectory for diff images if applicable
            if is_diff_images:
                diff_dir = supp_data_dir / "diff"
                diff_dir.mkdir(parents=True, exist_ok=True)
            else:
                diff_dir = None

            # Determine preprocessing type for filename suffix
            preproc_suffix = ""
            if self.config.get("zscore", False):
                preproc_suffix = "_preproc-zscore"
            elif self.config.get("scaling_key"):
                preproc_suffix = f"_preproc-scaled"

            for idx, img in enumerate(self._glm_input_images):
                if isinstance(img, nib.Nifti1Image):
                    # For in-memory images (z-scored/scaled/difference), generate BIDS-style filename
                    if is_diff_images:
                        # For paired analysis difference images
                        # Try to get subject info from paired samples
                        if hasattr(self, '_images') and idx < len(self._images) // 2:
                            # Get info from the first sample of the pair
                            img_info = self._images[idx]
                            filename = self._generate_data_filename(
                                img_info, idx, preproc_type="diff"
                            )
                        else:
                            filename = f"diff_{idx:04d}.nii.gz"
                        filepath = diff_dir / filename
                    elif hasattr(self, '_images') and idx < len(self._images):
                        # For z-scored/scaled images, generate filename with BIDS entities
                        img_info = self._images[idx]
                        preproc_type = "zscore" if self.config.get("zscore") else "scaled" if self.config.get("scaling_key") else None
                        filename = self._generate_data_filename(img_info, idx, preproc_type=preproc_type)
                        filepath = supp_data_dir / filename
                    else:
                        filename = f"input_{idx:04d}.nii.gz"
                        filepath = supp_data_dir / filename
                    nib.save(img, filepath)
                    saved_files[f"input_image_{idx}"] = filepath
                elif isinstance(img, (str, Path)):
                    # For path references, create symlinks with BIDS-style names
                    src_path = Path(img)
                    # Try to find matching image info
                    matching_info = None
                    if hasattr(self, '_images'):
                        for img_info in self._images:
                            if str(img_info['path']) == str(src_path):
                                matching_info = img_info
                                break

                    if matching_info:
                        filename = self._generate_data_filename(matching_info, idx, preproc_type=None)
                    else:
                        filename = src_path.name

                    if is_diff_images:
                        filepath = diff_dir / filename
                    else:
                        filepath = supp_data_dir / filename
                    # Create a symlink to avoid copying large files
                    if not filepath.exists():
                        try:
                            filepath.symlink_to(src_path.resolve())
                        except OSError:
                            # If symlink fails (e.g., cross-device), copy the file
                            import shutil
                            shutil.copy2(src_path, filepath)
                    saved_files[f"input_image_{idx}"] = filepath
            logger.info(f"  Saved {len(self._glm_input_images)} input images to supplementary_data/")
            print(f"  Saved {len(self._glm_input_images)} input images to supplementary_data/")

        # Save original unscaled images if they were normalized before analysis
        # Check both config and runtime parameters for scaling/zscore
        has_scaling = (
            self.config.get("scaling_key") or
            self.config.get("zscore") or
            (hasattr(self, '_runtime_scaling') and self._runtime_scaling is not None) or
            (hasattr(self, '_scaling_key') and self._scaling_key is not None)
        )

        original_images_to_save = None
        if (hasattr(self, '_original_paired_sample1') and
            hasattr(self, '_original_paired_sample2') and
            has_scaling):
            # Paired test with normalization
            original_images_to_save = self._original_paired_sample1 + self._original_paired_sample2
        elif (hasattr(self, '_original_valid_images') and
              has_scaling):
            # Two-sample or one-sample test with normalization
            original_images_to_save = self._original_valid_images

        if save_supp_data and original_images_to_save:
            logger.info(f"Saving {len(original_images_to_save)} original unscaled images...")
            print(f"\nSaving {len(original_images_to_save)} original unscaled images to supplementary_data/unscaled/")
            unscaled_dir = supp_data_dir / "unscaled"
            unscaled_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0
            for idx, img_info in enumerate(original_images_to_save):
                src_path = Path(img_info['path'])
                if not src_path.exists():
                    logger.warning(f"Source image not found: {src_path}")
                    continue

                filename = self._generate_data_filename(img_info, idx, preproc_type="unscaled")
                filepath = unscaled_dir / filename
                try:
                    if not filepath.exists():
                        filepath.symlink_to(src_path.resolve())
                    saved_files[f"unscaled_image_{idx}"] = filepath
                    saved_count += 1
                    logger.debug(f"  Saved/linked: {filepath}")
                except OSError as e:
                    # If symlink fails, try to copy the file
                    try:
                        import shutil
                        shutil.copy2(src_path, filepath)
                        saved_files[f"unscaled_image_{idx}"] = filepath
                        saved_count += 1
                        logger.debug(f"  Copied: {filepath}")
                    except Exception as copy_err:
                        logger.error(f"Failed to save {src_path}: {copy_err}")

            logger.info(f"  Saved {saved_count}/{len(original_images_to_save)} original images to {unscaled_dir}")
            print(f"  Saved {saved_count}/{len(original_images_to_save)} original unscaled images to supplementary_data/unscaled/")
        else:
            logger.info("No normalization applied or no original images to save")

        # Save masks used in z-scoring (for quality control and reproducibility)
        if save_supp_data and hasattr(self, '_zscore_masks_used') and self._zscore_masks_used:
            logger.info("Saving z-scoring masks used...")
            masks_log_dir = supp_data_dir / "zscore_masks"
            masks_log_dir.mkdir(parents=True, exist_ok=True)

            # Create a mapping file documenting which mask was used for each image
            masks_manifest_path = masks_log_dir / "mask_manifest.txt"
            with open(masks_manifest_path, "w") as f:
                f.write("# Masks used in z-scoring\n")
                f.write("# Image path -> Mask path\n\n")

                for img_path, mask_path in sorted(self._zscore_masks_used.items()):
                    f.write(f"{img_path} -> {mask_path}\n")

            saved_files["zscore_masks_manifest"] = masks_manifest_path
            logger.info(f"  Saved z-scoring mask manifest: {masks_manifest_path}")
            print(f"  Saved z-scoring mask manifest to supplementary_data/zscore_masks/")

            # Also create symlinks/copies to actual masks for convenience
            unique_masks = set(self._zscore_masks_used.values())
            for mask_path in unique_masks:
                mask_file = Path(mask_path)
                if mask_file.exists():
                    # Use original mask filename directly (preserves all BIDS entities)
                    # Masks have their own independent BIDS structure, don't derive from bids_prefix
                    symlink_name = mask_file.name
                    symlink_path = masks_log_dir / symlink_name

                    try:
                        if not symlink_path.exists():
                            symlink_path.symlink_to(mask_file.resolve())
                        saved_files[f"zscore_mask_{mask_file.stem}"] = symlink_path
                        logger.debug(f"  Created mask symlink: {symlink_path}")
                    except OSError:
                        # If symlink fails, try copying
                        try:
                            import shutil
                            shutil.copy2(mask_file, symlink_path)
                            saved_files[f"zscore_mask_{mask_file.stem}"] = symlink_path
                            logger.debug(f"  Copied mask: {symlink_path}")
                        except Exception as e:
                            logger.warning(f"Could not save mask {mask_path}: {e}")

        # Save statistical maps with BIDS naming and JSON sidecars
        if self.glm is not None:
            for contrast_name, result in self.glm.results.items():
                for map_type, stat_map in result["maps"].items():
                    # Create BIDS-compatible filename
                    filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{map_type}.nii.gz"
                    filepath = maps_dir / filename

                    nib.save(stat_map, filepath)
                    saved_files[f"{contrast_name}_{map_type}"] = filepath

                    # Create JSON sidecar
                    json_path = self._generate_json_sidecar(
                        filepath,
                        stat_type=map_type,
                        contrast_name=contrast_name,
                    )
                    saved_files[f"{contrast_name}_{map_type}_json"] = json_path

        # Save thresholded maps and cluster tables with BIDS naming
        if self.inference is not None:
            logger.info(f"Saving thresholded maps for corrections: {list(self.inference.thresholded_maps.keys())}")
            for contrast_name, corrections in self.inference.thresholded_maps.items():
                logger.info(f"  Contrast '{contrast_name}' has corrections: {list(corrections.keys())}")
                for correction, stat_map in corrections.items():
                    # Create BIDS-compatible filename
                    filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_threshold.nii.gz"
                    filepath = maps_dir / filename

                    nib.save(stat_map, filepath)
                    saved_files[f"{contrast_name}_{correction}_map"] = filepath
                    logger.info(f"  Saved {correction} map: {filepath}")

                    # Print for permutation results
                    if correction == "fwer_perm":
                        print(f"  Saved permutation result: maps/{filename}")

                    # Create JSON sidecar
                    threshold_value = None
                    if correction == "uncorrected":
                        threshold_value = self.inference.alpha_uncorrected
                    elif correction in ["fdr", "bonferroni"]:
                        threshold_value = self.inference.alpha_corrected
                    elif correction == "fwer_perm":
                        threshold_value = self.inference.alpha_corrected

                    # Get actual threshold from inference results
                    actual_threshold = None
                    if contrast_name in self.inference.threshold_values:
                        if correction in self.inference.threshold_values[contrast_name]:
                            actual_threshold = self.inference.threshold_values[contrast_name][correction]

                    json_path = self._generate_json_sidecar(
                        filepath,
                        stat_type="stat",
                        contrast_name=contrast_name,
                        threshold_type=correction,
                        threshold_value=threshold_value,
                        actual_threshold=actual_threshold,
                    )
                    saved_files[f"{contrast_name}_{correction}_map_json"] = json_path

            # Save cluster tables
            for contrast_name, corrections in self.inference.cluster_tables.items():
                for correction, table in corrections.items():
                    if len(table) == 0:
                        logger.info(f"  Skipping empty cluster table for {contrast_name}/{correction}")
                        continue

                    filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_clusters.tsv"
                    filepath = tables_dir / filename
                    table.to_csv(filepath, sep="\t", index=False)
                    saved_files[f"{contrast_name}_{correction}_table"] = filepath
                    logger.info(f"  Saved {correction} cluster table: {filepath}")

                    # Print for permutation results
                    if correction == "fwer_perm":
                        print(f"  Saved permutation cluster table: tables/{filename}")

            # Save enhanced cluster tables (if extra analysis was performed)
            if hasattr(self.inference, 'enhanced_cluster_tables'):
                for contrast_name, corrections in self.inference.enhanced_cluster_tables.items():
                    for correction, table in corrections.items():
                        if len(table) == 0:
                            continue

                        # Save enhanced cluster table
                        filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_clusters_enhanced.tsv"
                        filepath = tables_dir / filename
                        table.to_csv(filepath, sep="\t", index=False)
                        saved_files[f"{contrast_name}_{correction}_enhanced_table"] = filepath
                        logger.info(f"  Saved enhanced cluster table: {filepath}")
                        print(f"  Saved enhanced cluster table ({len(table)} clusters): tables/{filename}")

            # Save cortical/non-cortical summary tables (if extra analysis was performed)
            if hasattr(self.inference, 'cortical_summaries'):
                for contrast_name, corrections in self.inference.cortical_summaries.items():
                    for correction, summary in corrections.items():
                        if len(summary) == 0:
                            continue

                        # Save cortical/non-cortical summary
                        filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_cortical_summary.tsv"
                        filepath = tables_dir / filename
                        summary.to_csv(filepath, sep="\t", index=False)
                        saved_files[f"{contrast_name}_{correction}_cortical_summary"] = filepath
                        logger.info(f"  Saved cortical/non-cortical summary: {filepath}")
                        print(f"  Saved cortical/non-cortical summary: tables/{filename}")

            # Save permutation null distributions
            if self.inference is not None and hasattr(self.inference, '_inference_results'):
                for key, results in self.inference._inference_results.items():
                    if "_perm" in key and "h0_max_t" in results and results["h0_max_t"] is not None:
                        contrast_name = key.replace("_perm", "")
                        h0_dist = results["h0_max_t"]

                        # Save as numpy array in maps directory
                        filename = f"{bids_prefix}_contrast-{contrast_name}_stat-fwer_perm_h0distribution.npy"
                        filepath = maps_dir / filename
                        np.save(filepath, h0_dist)
                        saved_files[f"{contrast_name}_h0_distribution"] = filepath
                        logger.info(f"  Saved permutation null distribution: {filepath}")
                        print(f"  Saved null distribution ({len(h0_dist)} values): maps/{filename}")
        
        # Save design matrix to logs directory
        if self._design_matrix is not None:
            dm_path = logs_dir / "design_matrix.tsv"
            self._design_matrix.to_csv(dm_path, sep="\t", index=False)
            saved_files["design_matrix"] = dm_path

        # Save configuration with BIDS-compatible name matching the report
        # Include contrast name(s) in filename for easy reproducibility
        if self.glm is not None and self.glm.results:
            contrast_names = list(self.glm.results.keys())
            if len(contrast_names) == 1:
                # Single contrast: use specific name
                contrast_str = contrast_names[0]
            elif len(contrast_names) > 1:
                # Multiple contrasts: join with +
                contrast_str = "+".join(contrast_names)
            else:
                # No contrasts (shouldn't happen, but fallback)
                contrast_str = "none"
        else:
            # No GLM results yet (shouldn't happen in save_results)
            contrast_str = "none"

        # Create a complete config including runtime parameters for reproducibility
        complete_config = self._create_reproducible_config()

        # Save main config file in output directory with BIDS naming
        config_filename = f"{bids_prefix}_contrast-{contrast_str}_config.yaml"
        config_path = self.output_dir / config_filename

        # Save the complete config as YAML with header comment
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            # Write header with CLI command
            f.write("# ============================================================================\n")
            f.write("# StatCraft Configuration File\n")
            f.write("# ============================================================================\n")
            f.write("# This file contains the complete configuration for reproducing this analysis.\n")
            f.write("#\n")
            f.write("# OPTION 1: Rerun using this config file:\n")
            f.write(f"#   statcraft --config {config_filename}\n")
            f.write("#\n")
            f.write("# OPTION 2: Rerun using CLI arguments (equivalent command):\n")
            cli_cmd = complete_config.get('equivalent_cli_command', 'N/A')
            # Split long command into multiple lines for readability
            if len(cli_cmd) > 80:
                f.write("#   ")
                line_len = 4
                for part in cli_cmd.split():
                    if line_len + len(part) + 1 > 80:
                        f.write(" \\\n#     ")
                        line_len = 6
                    f.write(part + " ")
                    line_len += len(part) + 1
                f.write("\n")
            else:
                f.write(f"#   {cli_cmd}\n")
            f.write("#\n")
            f.write("# ============================================================================\n\n")

            # Use custom YAML dumper to format the CLI command nicely
            class CustomDumper(yaml.SafeDumper):
                pass

            def str_representer(dumper, data):
                # Use literal block style for multiline strings (preserves newlines and formatting)
                if '\n' in data:
                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return dumper.represent_scalar('tag:yaml.org,2002:str', data)

            CustomDumper.add_representer(str, str_representer)

            yaml.dump(complete_config, f, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)

        saved_files["config"] = config_path
        logger.info(f"  Saved analysis configuration: {config_path}")
        print(f"  Saved reproducible config: {config_filename}")
        print(f"  Equivalent CLI command saved in config file header")

        # Also save a copy in logs directory for backward compatibility
        logs_config_path = logs_dir / "config.yaml"
        with open(logs_config_path, "w") as f:
            # Write header with CLI command
            f.write("# ============================================================================\n")
            f.write("# StatCraft Configuration File\n")
            f.write("# ============================================================================\n")
            f.write("# This file contains the complete configuration for reproducing this analysis.\n")
            f.write("#\n")
            f.write("# OPTION 1: Rerun using this config file:\n")
            f.write(f"#   statcraft --config config.yaml\n")
            f.write("#\n")
            f.write("# OPTION 2: Rerun using CLI arguments (equivalent command):\n")
            cli_cmd = complete_config.get('equivalent_cli_command', 'N/A')
            # Split long command into multiple lines for readability
            if len(cli_cmd) > 80:
                f.write("#   ")
                line_len = 4
                for part in cli_cmd.split():
                    if line_len + len(part) + 1 > 80:
                        f.write(" \\\n#     ")
                        line_len = 6
                    f.write(part + " ")
                    line_len += len(part) + 1
                f.write("\n")
            else:
                f.write(f"#   {cli_cmd}\n")
            f.write("#\n")
            f.write("# ============================================================================\n\n")

            # Use custom YAML dumper to format the CLI command nicely
            class CustomDumper(yaml.SafeDumper):
                pass

            def str_representer(dumper, data):
                # Use literal block style for multiline strings (preserves newlines and formatting)
                if '\n' in data:
                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return dumper.represent_scalar('tag:yaml.org,2002:str', data)

            CustomDumper.add_representer(str, str_representer)

            yaml.dump(complete_config, f, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)

        # Save file lists to logs directory
        if self._all_images:
            # Save all discovered files
            all_files_path = logs_dir / "discovered_files.txt"
            with open(all_files_path, "w") as f:
                f.write(f"# All discovered files ({len(self._all_images)})\n")
                f.write(f"# Generated by StatCraft\n\n")
                for i, img in enumerate(self._all_images, 1):
                    f.write(f"{i}\t{img['path']}\n")
            saved_files["discovered_files"] = all_files_path

            # Save valid files
            valid_files_path = logs_dir / "valid_files.txt"
            with open(valid_files_path, "w") as f:
                f.write(f"# Valid files used in analysis ({len(self._images)})\n")
                f.write(f"# Generated by StatCraft\n\n")
                for i, img in enumerate(self._images, 1):
                    f.write(f"{i}\t{img['path']}\n")
            saved_files["valid_files"] = valid_files_path

            # Save excluded files if any
            if self._invalid_images:
                excluded_files_path = logs_dir / "excluded_files.txt"
                with open(excluded_files_path, "w") as f:
                    f.write(f"# Excluded files ({len(self._invalid_images)})\n")
                    f.write(f"# Generated by StatCraft\n\n")
                    for i, img in enumerate(self._invalid_images, 1):
                        reason = img.get('reason', 'Unknown')
                        f.write(f"{i}\t{img['path']}\t[{reason}]\n")
                saved_files["excluded_files"] = excluded_files_path

        logger.info(f"Saved {len(saved_files)} files to {self.output_dir}")
        return saved_files

    @staticmethod
    def _normalize_bids_entity_key(entity_key: str) -> str:
        """
        Normalize BIDS entity abbreviations to their full names.

        Parameters
        ----------
        entity_key : str
            Entity key (can be abbreviation or full name).

        Returns
        -------
        str
            Full entity name.
        """
        # Mapping from BIDS abbreviations to full names
        entity_mapping = {
            "sub": "subject",
            "ses": "session",
            "run": "run",
            "task": "task",
            "acq": "acquisition",
            "ce": "ceagent",
            "rec": "reconstruction",
            "dir": "direction",
            "mod": "modality",
            "echo": "echo",
            "flip": "flip",
            "inv": "inversion",
            "mt": "mt",
            "part": "part",
            "proc": "processing",
            "hemi": "hemisphere",
            "space": "space",
            "split": "split",
            "recording": "recording",
            "chunk": "chunk",
            "atlas": "atlas",
            "roi": "roi",
            "label": "label",
            "from": "from",
            "to": "to",
            "res": "resolution",
            "den": "density",
            "desc": "description",
            "stain": "stain",
        }

        # Return the full name if it's an abbreviation, otherwise return as-is
        return entity_mapping.get(entity_key, entity_key)

    def _find_mask_for_image(
        self,
        image_info: Dict,
        mask_pattern: str,
    ) -> Optional[Path]:
        """
        Find a mask file matching the mask pattern for a given image.

        Uses BIDS entity matching to find the correct mask when multiple masks
        match the pattern. Only enforces entity matching for entities that
        actually exist in the mask paths (e.g., won't require session match
        if masks don't have session information).

        Parameters
        ----------
        image_info : dict
            Image information dictionary with 'path' and 'entities'.
        mask_pattern : str
            Mask pattern or fixed path.

        Returns
        -------
        Path or None
            Path to matching mask, or None if not found.
        """
        from pathlib import Path
        import glob
        import os

        # Expand tilde to home directory
        mask_pattern = os.path.expanduser(mask_pattern)

        # Check if it's a fixed path (no wildcards)
        if '*' not in mask_pattern and '?' not in mask_pattern:
            mask_path = Path(mask_pattern)
            if mask_path.exists():
                return mask_path
            else:
                logger.warning(f"Fixed mask path does not exist: {mask_pattern}")
                return None

        # It's a pattern - do a glob to find all matching masks
        matches = list(glob.glob(mask_pattern))
        entities = image_info.get("entities", {})

        if len(matches) == 0:
            logger.warning(f"No mask found for image {image_info['path'].name} using pattern {mask_pattern}")
            return None

        if len(matches) == 1:
            # Single match - use it directly
            logger.debug(f"Found single mask using pattern: {matches[0]}")
            return Path(matches[0])

        # Multiple masks found - filter by BIDS entities
        logger.debug(f"Pattern {mask_pattern} matches {len(matches)} files. Filtering by BIDS entities...")

        # Filter matches based on image entities (subject, session, task)
        # Note: Only enforce entity match if the mask path/name contains that entity type
        filtered_matches = []
        for match in matches:
            match_path = Path(match)
            match_name = match_path.name
            match_str = str(match_path)

            # Check if this mask matches the image's key entities
            entity_match = True

            # Check subject - required match (subject should always be present)
            if 'subject' in entities:
                subject_str = f"sub-{entities['subject']}"
                if subject_str not in match_name and subject_str not in match_str:
                    entity_match = False

            # Check session - only require match if the mask has a session entity
            # (some masks like fmriprep brain masks may not have session in the path)
            if entity_match and 'session' in entities:
                session_str = f"ses-{entities['session']}"
                # Only filter by session if the mask path contains ANY session indicator
                if 'ses-' in match_str:
                    if session_str not in match_name and session_str not in match_str:
                        entity_match = False

            # Check task - only require match if the mask has a task entity
            if entity_match and 'task' in entities:
                task_str = f"task-{entities['task']}"
                # Only filter by task if the mask path contains ANY task indicator
                if 'task-' in match_str:
                    if task_str not in match_name and task_str not in match_str:
                        entity_match = False

            # Check run - only require match if the mask has a run entity
            if entity_match and 'run' in entities:
                run_str = f"run-{entities['run']}"
                # Only filter by run if the mask path contains ANY run indicator
                if 'run-' in match_str:
                    if run_str not in match_name and run_str not in match_str:
                        entity_match = False

            if entity_match:
                filtered_matches.append(match)

        if len(filtered_matches) == 1:
            logger.debug(f"Found unique mask after entity filtering: {filtered_matches[0]}")
            return Path(filtered_matches[0])
        elif len(filtered_matches) > 1:
            logger.warning(
                f"Multiple masks match image {image_info['path'].name} after filtering: "
                f"{len(filtered_matches)} matches. Using first: {filtered_matches[0]}"
            )
            return Path(filtered_matches[0])
        else:
            logger.warning(
                f"Pattern {mask_pattern} matches {len(matches)} files, "
                f"but none match the BIDS entities of {image_info['path'].name}. "
                "Check that masks and images have matching subject/session/task identifiers."
            )
            return None

    @staticmethod
    def _scale_image_by_mask(
        image: Union[str, Path, nib.Nifti1Image],
        mask: Union[str, Path, nib.Nifti1Image],
    ) -> nib.Nifti1Image:
        """
        Scale an image by dividing by the mean value within a mask.

        Parameters
        ----------
        image : str, Path, or nibabel image
            Image to scale.
        mask : str, Path, or nibabel image
            Mask defining the region for mean computation.

        Returns
        -------
        nibabel.Nifti1Image
            Scaled image.
        """
        # Load image with memory mapping
        if isinstance(image, (str, Path)):
            img = nib.load(image, mmap=True)
        else:
            img = image

        # Load mask with memory mapping
        if isinstance(mask, (str, Path)):
            mask_img = nib.load(mask, mmap=True)
        else:
            mask_img = mask

        # Get data as float32 to save memory
        img_data = np.asarray(img.dataobj, dtype=np.float32)
        mask_data = np.asarray(mask_img.dataobj, dtype=bool)

        # Compute mean within mask
        mask_voxels = mask_data > 0
        if not np.any(mask_voxels):
            raise ValueError("Mask is empty (no non-zero voxels)")

        mean_val = np.mean(img_data[mask_voxels])

        if mean_val == 0:
            raise ValueError("Mask region has zero mean - cannot scale by zero")

        if np.isnan(mean_val):
            raise ValueError("Mask region has NaN mean - check mask and image alignment")

        # Scale the data
        scaled_data = img_data / mean_val

        # Create new image
        scaled_img = nib.Nifti1Image(scaled_data, img.affine, img.header)

        logger.debug(f"Scaled image by mask mean: {mean_val:.4f}")

        return scaled_img

    def _apply_scaling(
        self,
        images_info: List[Dict],
        scaling_pattern: str,
    ) -> List[nib.Nifti1Image]:
        """
        Apply scaling to a list of images using masks.

        Parameters
        ----------
        images_info : list of dict
            List of image information dictionaries.
        scaling_pattern : str
            Mask pattern or fixed path.

        Returns
        -------
        list of nibabel.Nifti1Image
            List of scaled images.
        """
        import os

        logger.info(f"Applying data scaling with pattern: {scaling_pattern}")

        # Expand tilde to home directory
        scaling_pattern = os.path.expanduser(scaling_pattern)

        # Check if it's a fixed mask path
        is_fixed_mask = '*' not in scaling_pattern and '?' not in scaling_pattern
        fixed_mask = None

        if is_fixed_mask:
            fixed_mask_path = Path(scaling_pattern)
            if not fixed_mask_path.exists():
                raise FileNotFoundError(f"Fixed mask path does not exist: {scaling_pattern}")
            fixed_mask = nib.load(fixed_mask_path, mmap=True)
            logger.info(f"Using fixed mask for all images: {fixed_mask_path}")
        else:
            logger.info("Using per-subject/session mask matching")

        # Store scaling ROI paths used for each image (for saving to outputs)
        # Initialize dict only if it doesn't exist (don't reset on subsequent calls)
        if not hasattr(self, '_scaling_rois_used'):
            self._scaling_rois_used = {}

        # Prepare display list
        file_list = [f"[{idx}/{len(images_info)}] {img_info['path'].name}"
                     for idx, img_info in enumerate(images_info, 1)]
        _print_file_list_limited(file_list, prefix="  ")

        scaled_images = []
        n_scaled = 0

        for img_info in images_info:
            if is_fixed_mask:
                # Use the same mask for all images
                try:
                    scaled_img = self._scale_image_by_mask(img_info["path"], fixed_mask)
                    self._scaling_rois_used[str(img_info["path"])] = scaling_pattern
                    n_scaled += 1
                except Exception as e:
                    logger.error(f"Failed to scale image {img_info['path'].name}: {e}")
                    raise RuntimeError(f"Scaling failed for image {img_info['path'].name}: {e}")
            else:
                # Find specific mask for this image
                mask_path = self._find_mask_for_image(img_info, scaling_pattern)

                if mask_path is None:
                    error_msg = f"No mask found for image {img_info['path'].name} using pattern {scaling_pattern}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                try:
                    scaled_img = self._scale_image_by_mask(img_info["path"], mask_path)
                    self._scaling_rois_used[str(img_info["path"])] = str(mask_path)
                    n_scaled += 1
                except Exception as e:
                    logger.error(f"Failed to scale image {img_info['path'].name}: {e}")
                    raise RuntimeError(f"Scaling failed for image {img_info['path'].name}: {e}")

            scaled_images.append(scaled_img)

        logger.info(f"Scaling complete: {n_scaled} images scaled successfully")
        print(f"  Scaled {n_scaled}/{len(images_info)} images successfully")

        return scaled_images

    def _apply_zscore(
        self,
        images_info: List[Dict],
        mask_pattern: str,
    ) -> List[nib.Nifti1Image]:
        """
        Apply z-scoring to a list of images within brain mask: (x - mean(x)) / std(x).

        Parameters
        ----------
        images_info : list of dict
            List of image information dictionaries with 'path' keys.
        mask_pattern : str
            Brain mask pattern (e.g., '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz').
            Uses BIDS entity matching to find the correct mask for each image.

        Returns
        -------
        list of Nifti1Image
            Z-scored images.
        """
        import os
        import warnings
        from nilearn.image import resample_to_img

        logger.info(f"Applying z-scoring to images using mask pattern: {mask_pattern}")

        # Expand tilde to home directory
        mask_pattern = os.path.expanduser(mask_pattern)

        # Check if it's a fixed mask path
        is_fixed_mask = '*' not in mask_pattern and '?' not in mask_pattern
        fixed_mask_img = None

        if is_fixed_mask:
            fixed_mask_path = Path(mask_pattern)
            if not fixed_mask_path.exists():
                raise FileNotFoundError(f"Fixed mask path does not exist: {mask_pattern}")
            fixed_mask_img = nib.load(fixed_mask_path)
            logger.info(f"Using fixed mask for all images: {fixed_mask_path}")
            print(f"  Using fixed mask: {fixed_mask_path}")
        else:
            logger.info("Using per-subject/session mask matching")
            print("  Using per-subject/session mask matching")

        # Store mask paths used for each image (for saving to outputs)
        # Initialize dict only if it doesn't exist (don't reset on subsequent calls)
        if not hasattr(self, '_zscore_masks_used'):
            self._zscore_masks_used = {}

        # Prepare display list
        file_list = [f"[{idx}/{len(images_info)}] {img_info['path'].name}"
                     for idx, img_info in enumerate(images_info, 1)]
        _print_file_list_limited(file_list, prefix="  ")

        zscored_images = []
        n_zscored = 0

        # Cache for resampled masks (key: (mask_path, image_shape, image_affine_hash))
        resampled_mask_cache = {}

        for idx, img_info in enumerate(images_info, 1):
            img_path = img_info['path']
            logger.debug(f"Z-scoring image {idx}/{len(images_info)}: {img_path.name}")

            try:
                # Load image first to get its shape/affine for resampling
                img = nib.load(img_path)
                data = np.asarray(img.dataobj, dtype=np.float32)

                # Find the appropriate mask for this image
                if is_fixed_mask:
                    mask_img = fixed_mask_img
                    mask_path_used = mask_pattern
                else:
                    # Find specific mask for this image using BIDS entity matching
                    mask_path = self._find_mask_for_image(img_info, mask_pattern)

                    if mask_path is None:
                        error_msg = f"No mask found for image {img_info['path'].name} using pattern {mask_pattern}"
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)

                    mask_path_used = str(mask_path)
                    mask_img = nib.load(mask_path)

                # Log the mask being used
                logger.info(f"  [{idx}/{len(images_info)}] {img_path.name} -> mask: {Path(mask_path_used).name}")

                # Store the mask path used for this image (for saving to outputs)
                self._zscore_masks_used[str(img_path)] = mask_path_used

                # Check if mask needs resampling (different shape or affine)
                mask_shape = mask_img.shape[:3]
                img_shape = img.shape[:3]

                # Create a cache key based on mask path and target shape
                affine_hash = hash(img.affine.tobytes())
                cache_key = (mask_path_used, img_shape, affine_hash)

                if mask_shape != img_shape or not np.allclose(mask_img.affine, img.affine):
                    # Check cache first
                    if cache_key in resampled_mask_cache:
                        mask_data = resampled_mask_cache[cache_key]
                        logger.debug(f"Using cached resampled mask for {img_path.name}")
                    else:
                        logger.info(f"    Resampling mask from {mask_shape} to {img_shape}")
                        # Resample mask to image space using nearest neighbor (for binary masks)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=FutureWarning, module='nilearn')
                            resampled_mask = resample_to_img(
                                mask_img, img,
                                interpolation='nearest',
                                copy_header=True,
                                force_resample=True
                            )
                        mask_data = np.asarray(resampled_mask.dataobj, dtype=bool) > 0
                        # Cache the resampled mask
                        resampled_mask_cache[cache_key] = mask_data
                else:
                    mask_data = np.asarray(mask_img.dataobj, dtype=bool) > 0

                # Calculate mean and std only within the brain mask
                brain_voxels = mask_data > 0
                if not np.any(brain_voxels):
                    raise ValueError(f"Mask is empty for image {img_path.name}")

                brain_data = data[brain_voxels]
                mean_val = np.nanmean(brain_data)
                std_val = np.nanstd(brain_data)

                if std_val == 0:
                    logger.warning(f"Image has zero std within mask: {img_path.name}. Skipping z-scoring (will use zeros).")
                    zscored_data = np.zeros_like(data)
                else:
                    # Z-score: (x - mean) / std
                    # Apply to entire image but using mask-based statistics
                    zscored_data = (data - mean_val) / std_val
                    logger.debug(f"Z-scored image: mean={mean_val:.4f}, std={std_val:.4f} (computed within mask)")

                # Create new image with z-scored data
                zscored_img = nib.Nifti1Image(zscored_data, img.affine, img.header)
                zscored_images.append(zscored_img)
                n_zscored += 1

            except Exception as e:
                logger.error(f"Failed to z-score image {img_path.name}: {e}")
                raise RuntimeError(f"Z-scoring failed for image {img_path.name}: {e}")

        logger.info(f"Z-scoring complete: {n_zscored} images processed successfully")
        print(f"  Z-scored {n_zscored}/{len(images_info)} images successfully")

        return zscored_images

    def _run_paired_with_patterns(
        self,
        sample_patterns: Dict[str, str],
        exclude_pattern: Optional[Union[str, Dict[str, str]]] = None,
        scaling: Optional[str] = None,
        zscore: bool = False,
        mask: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run paired t-test using sample patterns.

        For each pair of samples, compute the difference image at the individual level,
        then perform a one-sample t-test on the difference images.

        Parameters
        ----------
        sample_patterns : dict
            Dictionary of sample_name -> pattern.
        exclude_pattern : str or dict, optional
            Pattern to exclude files. Can be a single pattern (applied to all samples)
            or a dict mapping sample names to their exclude patterns.
        scaling : str, optional
            Mask pattern or path for data scaling.
        zscore : bool, optional
            If True, z-score the data before analysis.
        mask : str, optional
            Brain mask pattern for z-scoring.

        Returns
        -------
        dict
            Dictionary with all results.
        """
        logger.info("Running paired t-test with sample patterns...")

        # Get sample names and patterns
        sample_names = list(sample_patterns.keys())
        if len(sample_names) != 2:
            raise ValueError(f"Paired t-test requires exactly 2 samples, got {len(sample_names)}")

        sample1_name, sample2_name = sample_names
        logger.info(f"Comparing {sample1_name} vs {sample2_name}")

        # Determine exclude patterns for each sample
        if isinstance(exclude_pattern, dict):
            exclude1 = exclude_pattern.get(sample1_name, None)
            exclude2 = exclude_pattern.get(sample2_name, None)
            print(f"\nUsing label-specific exclude patterns:")
            print(f"  {sample1_name}: {exclude1 if exclude1 else 'None'}")
            print(f"  {sample2_name}: {exclude2 if exclude2 else 'None'}")
        else:
            # Single pattern or None - apply to both
            exclude1 = exclude_pattern
            exclude2 = exclude_pattern
            if exclude_pattern:
                print(f"\nUsing global exclude pattern: {exclude_pattern}")

        # Load images for each sample
        sample1_images = self.data_loader.get_images(
            pattern=sample_patterns[sample1_name],
            exclude_pattern=exclude1,
        )
        sample2_images = self.data_loader.get_images(
            pattern=sample_patterns[sample2_name],
            exclude_pattern=exclude2,
        )

        logger.info(f"Found {len(sample1_images)} images for {sample1_name}")
        logger.info(f"Found {len(sample2_images)} images for {sample2_name}")

        # Print discovered files for each sample
        print(f"\n{sample1_name} files ({len(sample1_images)}):")
        print("=" * 80)
        sample1_display = []
        for i, img in enumerate(sample1_images, 1):
            entities_str = ""
            if 'entities' in img:
                entities = img['entities']
                parts = [f"{k}={v}" for k, v in sorted(entities.items())]
                entities_str = f" ({', '.join(parts)})"
            sample1_display.append(f"[{i:3d}] {img['path']}{entities_str}")
        _print_file_list_limited(sample1_display, prefix="  ")
        print("=" * 80)
        print()

        print(f"\n{sample2_name} files ({len(sample2_images)}):")
        print("=" * 80)
        sample2_display = []
        for i, img in enumerate(sample2_images, 1):
            entities_str = ""
            if 'entities' in img:
                entities = img['entities']
                parts = [f"{k}={v}" for k, v in sorted(entities.items())]
                entities_str = f" ({', '.join(parts)})"
            sample2_display.append(f"[{i:3d}] {img['path']}{entities_str}")
        _print_file_list_limited(sample2_display, prefix="  ")
        print("=" * 80)
        print()

        # Get pairing entity
        paired_config = self.config.get("paired_test", {})
        pair_by_input = paired_config.get("pair_by", "sub")

        # Normalize the entity key (e.g., "sub" -> "subject")
        pair_by = self._normalize_bids_entity_key(pair_by_input)

        if pair_by_input != "sub":
            logger.info(f"Pairing by BIDS entity: '{pair_by_input}' (normalized to '{pair_by}') [CUSTOM PAIRING]")
            print(f"\nPairing files by BIDS entity: '{pair_by_input}' (normalized to '{pair_by}') [CUSTOM PAIRING]")
        else:
            logger.info(f"Pairing by BIDS entity: '{pair_by_input}' (normalized to '{pair_by}') [DEFAULT]")
            print(f"\nPairing files by BIDS entity: '{pair_by_input}' (normalized to '{pair_by}') [DEFAULT]")
        print("=" * 80)

        # Create pairs based on the pairing entity
        sample1_by_entity = {}
        for img in sample1_images:
            if "entities" in img and pair_by in img["entities"]:
                entity_value = img["entities"][pair_by]
                sample1_by_entity[entity_value] = img
                logger.debug(f"{sample1_name}: {pair_by}={entity_value} -> {img['path']}")
            else:
                logger.warning(f"{sample1_name}: Image missing '{pair_by}' entity: {img['path']}")
                print(f"  WARNING: {sample1_name} image missing '{pair_by}' entity: {img['path']}")

        sample2_by_entity = {}
        for img in sample2_images:
            if "entities" in img and pair_by in img["entities"]:
                entity_value = img["entities"][pair_by]
                sample2_by_entity[entity_value] = img
                logger.debug(f"{sample2_name}: {pair_by}={entity_value} -> {img['path']}")
            else:
                logger.warning(f"{sample2_name}: Image missing '{pair_by}' entity: {img['path']}")
                print(f"  WARNING: {sample2_name} image missing '{pair_by}' entity: {img['path']}")

        # Find common entities
        sample1_entities = set(sample1_by_entity.keys())
        sample2_entities = set(sample2_by_entity.keys())
        common_entities = sample1_entities & sample2_entities

        # Report on entities found
        print(f"\n{sample1_name} entities found ({len(sample1_entities)}): {sorted(sample1_entities)}")
        print(f"{sample2_name} entities found ({len(sample2_entities)}): {sorted(sample2_entities)}")
        print(f"Common entities ({len(common_entities)}): {sorted(common_entities)}")

        # Report unpaired entities
        only_in_sample1 = sample1_entities - sample2_entities
        only_in_sample2 = sample2_entities - sample1_entities

        if only_in_sample1:
            print(f"\nWARNING: {len(only_in_sample1)} entities only in {sample1_name}: {sorted(only_in_sample1)}")
        if only_in_sample2:
            print(f"WARNING: {len(only_in_sample2)} entities only in {sample2_name}: {sorted(only_in_sample2)}")

        if not common_entities:
            print("\nERROR: No matching pairs found!")
            print(f"  - {sample1_name} has {len(sample1_entities)} unique '{pair_by_input}' values")
            print(f"  - {sample2_name} has {len(sample2_entities)} unique '{pair_by_input}' values")
            print(f"  - No overlap between the two sets")
            raise ValueError(f"No matching pairs found using '{pair_by_input}' entity")

        logger.info(f"Found {len(common_entities)} pairs using '{pair_by}' entity")
        print(f"\nSuccessfully paired {len(common_entities)} files:")
        print("=" * 80)

        # Validate pairs
        paired_sample1 = []
        paired_sample2 = []
        pair_display = []
        for entity in sorted(common_entities):
            paired_sample1.append(sample1_by_entity[entity])
            paired_sample2.append(sample2_by_entity[entity])
            logger.info(f"  Pair {pair_by}={entity}: {sample1_name} <-> {sample2_name}")
            pair_display.append(
                f"{pair_by_input}={entity}:\n"
                f"    {sample1_name}: {Path(sample1_by_entity[entity]['path']).name}\n"
                f"    {sample2_name}: {Path(sample2_by_entity[entity]['path']).name}"
            )

        _print_file_list_limited(pair_display, prefix="  ")
        print("=" * 80)
        print()

        # Validate MNI space for all images
        all_images = paired_sample1 + paired_sample2
        valid_images, invalid_images = self.data_loader.validate_mni_space(all_images)

        if invalid_images:
            logger.warning(f"Excluded {len(invalid_images)} invalid images")

        # Store for reporting
        self._all_images = all_images
        self._invalid_images = invalid_images

        # Store paired information for methodology section
        self._paired_info = {
            'pairs': pair_display,
            'sample1_name': sample1_name,
            'sample2_name': sample2_name,
            'pair_by': pair_by_input,
        }

        # Determine contrast order from config (before computing differences!)
        contrast_spec = self.config.get("contrast", f"{sample1_name}-{sample2_name}")
        logger.info(f"Contrast specification: {contrast_spec}")

        # Parse the contrast to determine order
        # Expected format: "A-B" means compute A minus B
        if "-" in contrast_spec:
            contrast_parts = contrast_spec.split("-")
            if len(contrast_parts) == 2:
                first_sample = contrast_parts[0].strip()
                second_sample = contrast_parts[1].strip()

                # Determine if we need to swap the order
                if first_sample == sample1_name and second_sample == sample2_name:
                    # Default order: sample1 - sample2
                    swap_order = False
                    logger.info(f"Computing differences as: {sample1_name} - {sample2_name}")
                elif first_sample == sample2_name and second_sample == sample1_name:
                    # Swapped order: sample2 - sample1
                    swap_order = True
                    logger.info(f"Computing differences as: {sample2_name} - {sample1_name}")
                else:
                    logger.warning(f"Contrast '{contrast_spec}' doesn't match sample names. Using default order.")
                    swap_order = False
            else:
                logger.warning(f"Invalid contrast format: '{contrast_spec}'. Expected 'A-B'. Using default order.")
                swap_order = False
        else:
            # No dash in contrast, use default order
            swap_order = False
            logger.info(f"No contrast specification, using default order: {sample1_name} - {sample2_name}")

        # Store original unscaled images for saving (before normalization)
        self._original_paired_sample1 = paired_sample1
        self._original_paired_sample2 = paired_sample2

        # Apply scaling or z-scoring if requested
        if scaling:
            print("\nApplying data scaling...")
            print("=" * 80)
            logger.info("Applying scaling to sample images before computing differences")

            # Scale sample1 images
            logger.info(f"Scaling {sample1_name} images...")
            print(f"\nScaling {sample1_name} images:")
            scaled_sample1_list = self._apply_scaling(paired_sample1, scaling)

            # Scale sample2 images
            logger.info(f"Scaling {sample2_name} images...")
            print(f"\nScaling {sample2_name} images:")
            scaled_sample2_list = self._apply_scaling(paired_sample2, scaling)

            # Create dict for quick lookup
            scaled_sample1 = {id(img): scaled_img for img, scaled_img in zip(paired_sample1, scaled_sample1_list)}
            scaled_sample2 = {id(img): scaled_img for img, scaled_img in zip(paired_sample2, scaled_sample2_list)}
            print("=" * 80)
            print()
        elif zscore:
            if mask is None:
                raise ValueError(
                    "A mask pattern is required for z-scoring. "
                    "Use --mask with a pattern like '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz'"
                )
            print("\nApplying z-scoring...")
            print("=" * 80)
            logger.info("Applying z-scoring to sample images before computing differences")

            # Z-score sample1 images
            logger.info(f"Z-scoring {sample1_name} images...")
            print(f"\nZ-scoring {sample1_name} images:")
            scaled_sample1_list = self._apply_zscore(paired_sample1, mask)

            # Z-score sample2 images
            logger.info(f"Z-scoring {sample2_name} images...")
            print(f"\nZ-scoring {sample2_name} images:")
            scaled_sample2_list = self._apply_zscore(paired_sample2, mask)

            # Create dict for quick lookup
            scaled_sample1 = {id(img): scaled_img for img, scaled_img in zip(paired_sample1, scaled_sample1_list)}
            scaled_sample2 = {id(img): scaled_img for img, scaled_img in zip(paired_sample2, scaled_sample2_list)}
            print("=" * 80)
            print()
        else:
            scaled_sample1 = None
            scaled_sample2 = None

        # Save scaled/z-scored images before computing differences
        if scaled_sample1 is not None and scaled_sample2 is not None:
            self._save_normalized_images(
                paired_sample1, paired_sample2,
                scaled_sample1_list, scaled_sample2_list,
                sample1_name, sample2_name,
                preproc_type="scaled" if scaling else "zscore"
            )

        # Compute difference images
        logger.info("Computing pairwise differences...")
        diff_images = []
        valid_paired_sample1 = []
        valid_paired_sample2 = []

        for img1, img2 in zip(paired_sample1, paired_sample2):
            # Check if both images are valid
            if img1 in invalid_images or img2 in invalid_images:
                logger.warning(f"Skipping invalid pair: {img1['path']} <-> {img2['path']}")
                continue

            valid_paired_sample1.append(img1)
            valid_paired_sample2.append(img2)

            # Load or use scaled/zscored images
            if scaling or zscore:
                nii1 = scaled_sample1[id(img1)]
                nii2 = scaled_sample2[id(img2)]
            else:
                nii1 = nib.load(img1["path"], mmap=True)
                nii2 = nib.load(img2["path"], mmap=True)

            # Compute difference in the correct order based on contrast specification
            if swap_order:
                # Compute sample2 - sample1
                diff_data = np.asarray(nii2.dataobj, dtype=np.float32) - np.asarray(nii1.dataobj, dtype=np.float32)
                diff_img = nib.Nifti1Image(diff_data, nii2.affine, nii2.header)
            else:
                # Compute sample1 - sample2 (default)
                diff_data = np.asarray(nii1.dataobj, dtype=np.float32) - np.asarray(nii2.dataobj, dtype=np.float32)
                diff_img = nib.Nifti1Image(diff_data, nii1.affine, nii1.header)
            diff_images.append(diff_img)

        logger.info(f"Computed {len(diff_images)} difference images")

        # Store difference images for permutation testing
        self._diff_images = diff_images

        # Store valid images for reporting
        # Use BOTH samples for BIDS prefix generation to exclude sample-specific entities
        self._images = valid_paired_sample1 + valid_paired_sample2
        self._image_paths = [str(img["path"]) for img in self._images]

        # Perform one-sample t-test on difference images
        logger.info("Performing one-sample t-test on difference images...")

        # Get contrast name from config - this should already reflect the actual order
        # since we parsed it above to determine swap_order
        if swap_order:
            default_contrast = f"{sample2_name}_minus_{sample1_name}"
        else:
            default_contrast = f"{sample1_name}_minus_{sample2_name}"

        contrast_name = self.config.get("contrast", default_contrast)
        # Ensure consistent naming: replace dashes with "minus" for clarity
        if "-" in contrast_name:
            contrast_name = contrast_name.replace(" ", "").replace("-", "_minus_")

        # Run one-sample t-test
        # Get smoothing_fwhm from config
        smoothing_fwhm = self.config.get("glm", {}).get("smoothing_fwhm", 0)
        self.glm = SecondLevelGLM(smoothing_fwhm=smoothing_fwhm)
        results = self.glm.one_sample_ttest(diff_images, contrast_name=contrast_name)

        # Store input images (difference images) for saving to data/ folder
        self._glm_input_images = diff_images

        # Build a simple design matrix for reporting
        self._design_matrix = pd.DataFrame({
            "intercept": np.ones(len(diff_images)),
        })

        # Add contrasts for reporting
        from statcraft.core.design_matrix import DesignMatrixBuilder
        self.design_matrix_builder = DesignMatrixBuilder(self._images)
        self.design_matrix_builder.contrasts = {contrast_name: np.array([1.0])}

        # Run inference
        logger.info("Running statistical inference...")
        self.run_inference()

        # Annotate clusters
        logger.info("Annotating clusters...")
        self.annotate_clusters()

        # Generate report
        output_config = self.config.get("output", {})
        if output_config.get("generate_report", True):
            logger.info("Generating report...")
            report_path = self.generate_report()
            logger.info(f"Report saved: {report_path}")

        # Save results
        logger.info("Saving results...")
        saved_files = self.save_results()

        logger.info("✓ Paired t-test with patterns completed successfully")

        return {
            "images": self._images,
            "design_matrix": self._design_matrix,
            "contrasts": self.design_matrix_builder.contrasts,
            "glm_results": self.glm.results,
            "cluster_tables": self.inference.cluster_tables,
            "saved_files": saved_files,
        }

    def _run_two_sample_with_patterns(
        self,
        sample_patterns: Dict[str, str],
        exclude_pattern: Optional[Union[str, Dict[str, str]]] = None,
        scaling: Optional[str] = None,
        zscore: bool = False,
        mask: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run two-sample t-test using sample patterns.

        Parameters
        ----------
        sample_patterns : dict
            Dictionary of sample_name -> pattern.
        exclude_pattern : str or dict, optional
            Pattern to exclude files. Can be a single pattern (applied to all samples)
            or a dict mapping sample names to their exclude patterns.
        scaling : str, optional
            Mask pattern or path for data scaling.
        zscore : bool, optional
            If True, z-score the data before analysis.
        mask : str, optional
            Brain mask pattern for z-scoring.

        Returns
        -------
        dict
            Dictionary with all results.
        """
        logger.info("Running two-sample t-test with sample patterns...")

        # Print exclude pattern info if using label-specific excludes
        if isinstance(exclude_pattern, dict):
            print(f"\nUsing label-specific exclude patterns:")
            for sample_name in sample_patterns.keys():
                exclude = exclude_pattern.get(sample_name, None)
                print(f"  {sample_name}: {exclude if exclude else 'None'}")
        elif exclude_pattern:
            print(f"\nUsing global exclude pattern: {exclude_pattern}")

        # Load images for each sample
        sample_images = {}
        for sample_name, pattern in sample_patterns.items():
            # Determine exclude pattern for this sample
            if isinstance(exclude_pattern, dict):
                exclude = exclude_pattern.get(sample_name, None)
            else:
                exclude = exclude_pattern

            images = self.data_loader.get_images(
                pattern=pattern,
                exclude_pattern=exclude,
            )
            logger.info(f"Found {len(images)} images for sample '{sample_name}'")
            sample_images[sample_name] = images

        # Combine all images
        all_images = []
        for images in sample_images.values():
            all_images.extend(images)

        # Validate MNI space
        valid_images, invalid_images = self.data_loader.validate_mni_space(all_images)

        if not valid_images:
            raise ValueError("No valid images after MNI space validation")

        if invalid_images:
            logger.warning(f"Excluded {len(invalid_images)} invalid images")

        # Store for reporting
        self._all_images = all_images
        self._invalid_images = invalid_images

        # Split valid images by sample
        valid_sample_images = {}
        for sample_name, images in sample_images.items():
            valid_sample_images[sample_name] = [img for img in images if img in valid_images]
            logger.info(f"Valid images for '{sample_name}': {len(valid_sample_images[sample_name])}")

        # Store all valid images
        self._images = valid_images
        self._image_paths = [str(img["path"]) for img in valid_images]

        # Store original images before normalization (for saving unscaled versions)
        self._original_valid_images = valid_images
        self._original_valid_sample_images = valid_sample_images

        # Apply scaling or z-scoring if requested
        normalized_images = None
        if scaling:
            print("\nApplying data scaling...")
            print("=" * 80)
            logger.info("Applying scaling to images before GLM fitting")
            scaled_images = self._apply_scaling(valid_images, scaling)
            # Replace the image paths with scaled images
            self._image_paths = scaled_images
            normalized_images = scaled_images
            print("=" * 80)
            print()
        elif zscore:
            if mask is None:
                raise ValueError(
                    "A mask pattern is required for z-scoring. "
                    "Use --mask with a pattern like '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz'"
                )
            print("\nApplying z-scoring...")
            print("=" * 80)
            logger.info("Applying z-scoring to images before GLM fitting")
            zscored_images = self._apply_zscore(valid_images, mask)
            # Replace the image paths with z-scored images
            self._image_paths = zscored_images
            normalized_images = zscored_images
            print("=" * 80)
            print()

        # Save normalized images before GLM fitting
        if normalized_images is not None:
            self._save_normalized_images_two_sample(
                valid_images, normalized_images,
                preproc_type="scaled" if scaling else "zscore"
            )

        # Build design matrix with sample indicators
        logger.info("Building design matrix...")
        sample_names = list(sample_patterns.keys())

        # Create design matrix
        design_data = {sample_name: [] for sample_name in sample_names}
        for img in valid_images:
            for sample_name in sample_names:
                if img in valid_sample_images[sample_name]:
                    design_data[sample_name].append(1.0)
                else:
                    design_data[sample_name].append(0.0)

        self._design_matrix = pd.DataFrame(design_data)

        # Fit GLM
        logger.info("Fitting GLM...")
        # Get smoothing_fwhm from config
        smoothing_fwhm = self.config.get("glm", {}).get("smoothing_fwhm", 0)
        self.glm = SecondLevelGLM(smoothing_fwhm=smoothing_fwhm)
        self.glm.fit(self._image_paths, self._design_matrix)

        # Store input images for saving to data/ folder
        self._glm_input_images = self._image_paths

        # Parse and compute contrast
        contrast_expr = self.config.get("contrast")
        if not contrast_expr:
            # Default contrast: first sample - second sample
            contrast_expr = f"{sample_names[0]} - {sample_names[1]}"

        logger.info(f"Computing contrast: {contrast_expr}")

        # Use nilearn's expression_to_contrast_vector to parse the contrast
        from nilearn.glm import expression_to_contrast_vector
        try:
            contrast_vector = expression_to_contrast_vector(
                contrast_expr,
                list(self._design_matrix.columns)
            )
            logger.info(f"Contrast vector: {contrast_vector}")
        except Exception as e:
            logger.error(f"Failed to parse contrast expression '{contrast_expr}': {e}")
            raise ValueError(f"Invalid contrast expression: {contrast_expr}")

        # Compute contrast
        # Ensure consistent naming: replace dashes with "minus" for clarity
        contrast_name = contrast_expr.replace(" ", "").replace("-", "_minus_")
        results = self.glm.compute_contrast(
            contrast=contrast_vector,
            contrast_name=contrast_name,
        )

        # Build design matrix builder for reporting
        from statcraft.core.design_matrix import DesignMatrixBuilder
        self.design_matrix_builder = DesignMatrixBuilder(self._images)
        self.design_matrix_builder.contrasts = {contrast_name: np.array(contrast_vector)}

        # Run inference
        logger.info("Running statistical inference...")
        self.run_inference()

        # Annotate clusters
        logger.info("Annotating clusters...")
        self.annotate_clusters()

        # Generate report
        output_config = self.config.get("output", {})
        if output_config.get("generate_report", True):
            logger.info("Generating report...")
            report_path = self.generate_report()
            logger.info(f"Report saved: {report_path}")

        # Save results
        logger.info("Saving results...")
        saved_files = self.save_results()

        logger.info("✓ Two-sample t-test with patterns completed successfully")

        return {
            "images": self._images,
            "design_matrix": self._design_matrix,
            "contrasts": self.design_matrix_builder.contrasts,
            "glm_results": self.glm.results,
            "cluster_tables": self.inference.cluster_tables,
            "saved_files": saved_files,
        }

    def run(
        self,
        participant_label: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        exclude_pattern: Optional[Union[str, Dict[str, str]]] = None,
        sample_patterns: Optional[Dict[str, str]] = None,
        scaling: Optional[str] = None,
        scaling_key: Optional[str] = None,
        zscore: bool = False,
        mask: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Parameters
        ----------
        participant_label : list of str, optional
            List of participant labels to include (without 'sub-' prefix).
        pattern : str, optional
            Glob pattern for finding files (for one-sample tests).
        exclude_pattern : str or dict, optional
            Glob pattern to exclude files from the match. Can be a single pattern
            (applied to all samples) or a dict mapping sample names to their
            exclude patterns (when using sample_patterns).
        sample_patterns : dict, optional
            Dictionary of sample_name -> pattern for multi-sample analyses.
        scaling : str, optional
            Mask pattern or path for data scaling. If contains wildcards,
            finds per-subject/session masks. If fixed path, uses same mask
            for all images. Mutually exclusive with zscore.
        scaling_key : str, optional
            Key to identify the scaling method in output filenames.
        zscore : bool, optional
            If True, z-score individual data at the subject level before group
            analysis: (x - mean(x)) / std(x). Mutually exclusive with scaling.
            Appears as '_scaling-{key}_' in output files.
        mask : str, optional
            Brain mask pattern for z-scoring. Required when zscore=True.
            Format: '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz'.
            Uses entity matching to find the correct mask for each image.

        Returns
        -------
        dict
            Dictionary with all results.
        """
        logger.info("Starting StatCraft pipeline...")

        # Store runtime parameters for reproducibility
        self._scaling_key = scaling_key if scaling else ("zscore" if zscore else None)
        self._runtime_pattern = pattern
        self._runtime_exclude_pattern = exclude_pattern
        self._runtime_sample_patterns = sample_patterns
        self._runtime_scaling = scaling
        self._runtime_zscore = zscore
        self._runtime_bids_filters = bids_filters

        analysis_type = self.config.get("analysis_type", "glm")

        # Get sample_patterns from config if not provided as parameter
        if sample_patterns is None:
            # Check if sample_patterns is in the config
            if analysis_type == "paired":
                sample_patterns = self.config.get("paired_test", {}).get("sample_patterns")
            elif analysis_type == "two-sample":
                sample_patterns = self.config.get("sample_patterns")

            # Update runtime parameter if we got it from config
            if sample_patterns is not None:
                logger.info(f"Using sample_patterns from config: {list(sample_patterns.keys())}")
                self._runtime_sample_patterns = sample_patterns

        # Get pattern from config if not provided as parameter
        if pattern is None:
            pattern = self.config.get("file_pattern")
            if pattern is not None:
                logger.info(f"Using file_pattern from config: {pattern}")
                self._runtime_pattern = pattern

        # Get exclude_pattern from config if not provided as parameter
        if exclude_pattern is None:
            exclude_pattern = self.config.get("exclude_pattern")
            if exclude_pattern is not None:
                if isinstance(exclude_pattern, dict):
                    logger.info(f"Using label-specific exclude_pattern from config: {list(exclude_pattern.keys())}")
                else:
                    logger.info(f"Using exclude_pattern from config: {exclude_pattern}")
                self._runtime_exclude_pattern = exclude_pattern

        # Get scaling from config if not provided as parameter
        if scaling is None:
            scaling = self.config.get("scaling_pattern")
            if scaling is not None:
                logger.info(f"Using scaling_pattern from config: {scaling}")
                self._runtime_scaling = scaling

        # Get scaling_key from config if not provided as parameter
        if scaling_key is None:
            scaling_key = self.config.get("scaling_key")
            if scaling_key is not None:
                logger.info(f"Using scaling_key from config: {scaling_key}")
                self._scaling_key = scaling_key

        # Get mask pattern from config if not provided as parameter (used with zscore)
        if mask is None:
            mask = self.config.get("mask_pattern")
            if mask is not None:
                logger.info(f"Using mask_pattern from config: {mask}")

        # Check if we're using multi-sample patterns
        if sample_patterns:
            if analysis_type == "paired":
                return self._run_paired_with_patterns(sample_patterns, exclude_pattern, scaling, zscore, mask)
            elif analysis_type == "two-sample":
                return self._run_two_sample_with_patterns(sample_patterns, exclude_pattern, scaling, zscore, mask)
            else:
                raise ValueError(f"sample_patterns only supported for paired and two-sample analyses, not {analysis_type}")

        # Standard single-pattern pipeline
        # Step 1: Load data
        self.load_data(participant_label=participant_label, pattern=pattern, exclude_pattern=exclude_pattern)

        # Store original images before normalization (for saving unscaled versions)
        self._original_valid_images = self._images.copy() if hasattr(self, '_images') else None

        # Step 1.5: Apply scaling or z-scoring if requested
        normalized_images = None
        if scaling:
            print("\nApplying data scaling...")
            print("=" * 80)
            logger.info("Applying scaling to images before GLM fitting")
            scaled_images = self._apply_scaling(self._images, scaling)
            # Replace the image paths with scaled images
            self._image_paths = scaled_images
            normalized_images = scaled_images
            print("=" * 80)
            print()
        elif zscore:
            if mask is None:
                raise ValueError(
                    "A mask pattern is required for z-scoring. "
                    "Use --mask with a pattern like '/path/to/fmriprep/sub-*/*/*brain*mask.nii.gz'"
                )
            print("\nApplying z-scoring...")
            print("=" * 80)
            logger.info("Applying z-scoring to images before GLM fitting")
            zscored_images = self._apply_zscore(self._images, mask)
            # Replace the image paths with z-scored images
            self._image_paths = zscored_images
            normalized_images = zscored_images
            print("=" * 80)
            print()

        # Save normalized images before GLM fitting
        if normalized_images is not None:
            self._save_normalized_images_two_sample(
                self._images, normalized_images,
                preproc_type="scaled" if scaling else "zscore"
            )

        # Step 2: Build design matrix
        self.build_design_matrix()

        # Step 3: Add contrasts
        self.add_contrasts()

        # Step 4: Fit model
        self.fit_model()
        
        # Step 5: Run inference
        self.run_inference()
        
        # Step 6: Annotate clusters
        self.annotate_clusters()
        
        # Step 7: Save results
        saved_files = self.save_results()
        
        # Step 8: Generate report
        output_config = self.config.get("output", {})
        if output_config.get("generate_report", True):
            report_path = self.generate_report()
            saved_files["report"] = report_path
        
        logger.info("Pipeline completed successfully!")
        
        return {
            "images": self._images,
            "design_matrix": self._design_matrix,
            "contrasts": self.design_matrix_builder.contrasts,
            "glm_results": self.glm.results,
            "cluster_tables": self.inference.cluster_tables,
            "saved_files": saved_files,
        }

    def run_connectivity_analysis(
        self,
        participant_label: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        sample_patterns: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run connectivity matrix analysis pipeline.

        This method handles edge-wise GLM analysis on connectivity matrices,
        with FDR/Bonferroni correction for multiple comparisons.

        Parameters
        ----------
        participant_label : list of str, optional
            List of participant labels to include.
        pattern : str, optional
            Glob pattern for finding .npy files (default: '**/*.npy').
        exclude_pattern : str, optional
            Glob pattern to exclude files.
        sample_patterns : dict, optional
            Dictionary of sample_name -> pattern for multi-sample analyses.

        Returns
        -------
        dict
            Dictionary with all results.
        """
        logger.info("Starting connectivity analysis pipeline...")

        analysis_type = self.config.get("analysis_type", "one-sample")

        # Set default pattern for .npy files
        if pattern is None:
            pattern = self.config.get("file_pattern", "**/*.npy")

        # Load connectivity data
        self.load_connectivity_data(
            participant_label=participant_label,
            pattern=pattern,
            exclude_pattern=exclude_pattern,
        )

        # Build design matrix
        self.build_design_matrix()

        # Add contrasts
        self.add_contrasts()

        # Load all connectivity matrices
        logger.info("Loading connectivity matrices...")
        matrices = []
        for img in self._images:
            matrix = self.data_loader.load_connectivity_matrix(img['path'])
            matrices.append(matrix)
        logger.info(f"Loaded {len(matrices)} matrices of shape {matrices[0].shape}")

        # Initialize connectivity GLM
        self.connectivity_glm = ConnectivityGLM()

        # Fit GLM based on analysis type
        if analysis_type == "one-sample":
            contrast_name = self.config.get("contrast_name", "one_sample")
            results = self.connectivity_glm.one_sample_ttest(
                matrices,
                contrast_name=contrast_name,
            )
        elif analysis_type == "two-sample":
            # Split matrices by group
            group_col = self.config.get("group_column", "group")
            if group_col not in self._design_matrix.columns:
                raise ValueError(f"Group column '{group_col}' not found in design matrix")

            # Get group labels
            groups = self._design_matrix[group_col].unique()
            if len(groups) != 2:
                raise ValueError(f"Expected 2 groups for two-sample test, got {len(groups)}")

            group1, group2 = sorted(groups)
            idx1 = self._design_matrix[group_col] == group1
            idx2 = self._design_matrix[group_col] == group2

            matrices_g1 = [m for m, keep in zip(matrices, idx1) if keep]
            matrices_g2 = [m for m, keep in zip(matrices, idx2) if keep]

            contrast_name = f"{group1}_vs_{group2}"
            results = self.connectivity_glm.two_sample_ttest(
                matrices_g1, matrices_g2,
                group1_name=str(group1),
                group2_name=str(group2),
                contrast_name=contrast_name,
            )
        elif analysis_type == "paired":
            # Get paired test config
            paired_config = self.config.get("paired_test", {})
            cond1 = paired_config.get("condition1", "condition1")
            cond2 = paired_config.get("condition2", "condition2")

            # Split matrices by condition
            n_subjects = len(matrices) // 2
            matrices_c1 = matrices[:n_subjects]
            matrices_c2 = matrices[n_subjects:]

            contrast_name = f"{cond1}_vs_{cond2}"
            results = self.connectivity_glm.paired_ttest(
                matrices_c1, matrices_c2,
                condition1_name=cond1,
                condition2_name=cond2,
                contrast_name=contrast_name,
            )
        else:
            # GLM analysis - compute all contrasts
            self.connectivity_glm.fit(matrices, self._design_matrix)
            for contrast_name, contrast_vector in self.design_matrix_builder.contrasts.items():
                results = self.connectivity_glm.compute_contrast(
                    contrast_vector,
                    contrast_name=contrast_name,
                )

        # Run inference for all contrasts
        logger.info("Running statistical inference on edges...")
        self.connectivity_inference = ConnectivityInference(
            alpha_corrected=self.config.get("alpha_corrected", 0.05),
            alpha_uncorrected=self.config.get("alpha_uncorrected", 0.001),
            two_sided=self.config.get("two_sided", True),
        )

        # Apply corrections
        corrections = self.config.get("corrections", ["fdr", "bonferroni"])
        
        # Add permutation testing if requested via inference config
        inf_config = self.config.get("inference", {})
        if inf_config.get("run_permutation", False):
            # Add fwer_perm to corrections if not already present
            if "fwer_perm" not in corrections and "permutation" not in corrections:
                corrections = list(corrections) + ["fwer_perm"]
                logger.info(f"Added permutation testing to corrections: {corrections}")
        
        # Run inference for each contrast
        for contrast_name, glm_result in self.connectivity_glm.results.items():
            t_matrix = glm_result['maps']['t_matrix']
            p_matrix = glm_result['maps']['p_matrix']
            df = glm_result['df']
            
            for correction in corrections:
                if correction == "fdr":
                    self.connectivity_inference.threshold_fdr(
                        t_matrix, p_matrix, df,
                        contrast_name=contrast_name,
                    )
                elif correction == "bonferroni":
                    self.connectivity_inference.threshold_bonferroni(
                        t_matrix, p_matrix, df,
                        contrast_name=contrast_name,
                    )
                elif correction == "uncorrected":
                    self.connectivity_inference.threshold_uncorrected(
                        t_matrix, p_matrix, df,
                        contrast_name=contrast_name,
                    )
                elif correction in ["fwer_perm", "permutation"]:
                    # Permutation testing for connectivity using nilearn's permuted_ols
                    logger.info(f"Running permutation testing for connectivity analysis ({contrast_name})...")
                    n_perm = inf_config.get("n_permutations", 1000)
                    n_jobs = inf_config.get("n_jobs", 1)
                    random_state = inf_config.get("random_state", None)
                    
                    # Get vectorized connectivity data from the GLM
                    connectivity_data = self.connectivity_glm._matrices
                    if connectivity_data is None:
                        raise ValueError("Connectivity GLM has no stored data. Cannot run permutation testing.")
                    
                    try:
                        self.connectivity_inference.threshold_fwer_permutation(
                            t_matrix, p_matrix, df,
                            connectivity_data=connectivity_data,
                            design_matrix=self._design_matrix.values if self._design_matrix is not None else np.array([]),
                            contrast_vector=self.design_matrix_builder.contrasts.get(contrast_name, np.array([])) if self.design_matrix_builder else np.array([]),
                            n_perm=n_perm,
                            contrast_name=contrast_name,
                            n_jobs=n_jobs,
                            random_state=random_state,
                        )
                        logger.info(f"✓ Permutation test completed successfully for connectivity ({contrast_name})")
                    except Exception as e:
                        logger.warning(f"Permutation test failed for connectivity ({contrast_name}): {e}")
                        import traceback
                        traceback.print_exc()


        # Save results with BIDS naming
        logger.info("Saving connectivity results...")
        
        # Generate BIDS prefix for connectivity analysis
        bids_prefix = self._generate_bids_prefix()
        logger.info(f"Using BIDS prefix for connectivity: {bids_prefix}")

        # Save GLM results (creates maps/ subdirectory)
        saved_files = self.connectivity_glm.save_results(
            self.output_dir,
            bids_prefix=bids_prefix,
            roi_names=self._connectivity_metadata.get('roi_names'),
            roi_coordinates=self._connectivity_metadata.get('roi_coordinates'),
            atlas_name=self._connectivity_metadata.get('atlas_name'),
        )

        # Save inference results (creates maps/ and tables/ subdirectories)
        saved_files.update(self.connectivity_inference.save_results(
            self.output_dir,
            bids_prefix=bids_prefix,
        ))

        # Generate report
        output_config = self.config.get("output", {})
        if output_config.get("generate_report", True):
            # Generate reports for all contrasts
            contrast_names = list(self.connectivity_glm.results.keys())
            report_path = self._generate_connectivity_report(contrast_names)
            saved_files["report"] = report_path

        logger.info("Connectivity analysis completed successfully!")

        return {
            "images": self._images,
            "design_matrix": self._design_matrix,
            "glm_results": self.connectivity_glm.results,
            "edge_tables": self.connectivity_inference.edge_tables,
            "saved_files": saved_files,
        }

    def _generate_connectivity_report(self, contrast_names: Union[str, List[str]]) -> Path:
        """
        Generate HTML report for connectivity analysis.

        Parameters
        ----------
        contrast_names : str or list of str
            Name(s) of the contrast(s) analyzed.

        Returns
        -------
        Path
            Path to saved report.
        """
        # Convert single contrast to list
        if isinstance(contrast_names, str):
            contrast_names = [contrast_names]
        
        logger.info(f"Generating connectivity analysis report for {len(contrast_names)} contrast(s)...")

        # Initialize report
        title = f"StatCraft Connectivity Analysis"
        if len(contrast_names) == 1:
            title = f"{title}: {contrast_names[0]}"
        
        self.report = ReportGenerator(
            title=title,
            output_dir=self.output_dir,
        )

        # Add methodology section
        self.report.add_section(
            "Analysis Overview",
            f"""
            <p><strong>Data Type:</strong> Connectivity Matrices</p>
            <p><strong>Number of Subjects:</strong> {len(self._images)}</p>
            <p><strong>Number of ROIs:</strong> {self.connectivity_glm.n_rois}</p>
            <p><strong>Number of Edges:</strong> {self.connectivity_glm.n_edges}</p>
            <p><strong>Atlas:</strong> {self._connectivity_metadata.get('atlas_name', 'Not specified')}</p>
            <p><strong>Contrasts:</strong> {', '.join(contrast_names)}</p>
            """,
            section_type="html",
            level=1,
        )

        # Add design matrix
        if self._design_matrix is not None:
            fig_key = self.report._plot_design_matrix(self._design_matrix)
            self.report.add_section(
                "Design Matrix",
                f'<div class="figure"><img src="data:image/png;base64,{self.report._figures[fig_key]}" alt="Design Matrix"></div>',
                section_type="html",
                level=2,
            )

        if self.connectivity_inference is None:
            logger.error("Connectivity inference not initialized")
            raise RuntimeError("Connectivity inference not initialized")

        # Add results for each contrast
        for contrast_name in contrast_names:
            # Add contrast header if multiple contrasts
            if len(contrast_names) > 1:
                self.report.add_section(
                    f"Contrast: {contrast_name}",
                    "",
                    section_type="html",
                    level=2,
                )
            
            glm_result = self.connectivity_glm.results[contrast_name]
            t_matrix_unthresholded = glm_result['maps']['t_matrix']
            p_matrix = glm_result['maps']['p_matrix']
            df = glm_result['df']

            for i, (correction, threshold) in enumerate(self.connectivity_inference.threshold_values.get(contrast_name, {}).items()):
                edge_table = self.connectivity_inference.edge_tables.get(contrast_name, {}).get(correction)
                # Get the thresholded matrix for this correction method
                thresholded_dict = self.connectivity_inference.thresholded_matrices.get(contrast_name, {})
                if correction in thresholded_dict:
                    t_matrix_thresholded = thresholded_dict[correction]
                else:
                    t_matrix_thresholded = t_matrix_unthresholded

                # Debug: log matrix properties
                n_nonzero = np.count_nonzero(t_matrix_thresholded)
                nonzero_vals = t_matrix_thresholded[t_matrix_thresholded != 0]
                if len(nonzero_vals) > 0:
                    logger.info(f"Report {correction}: {n_nonzero} non-zero edges, "
                               f"range=[{nonzero_vals.min():.3f}, {nonzero_vals.max():.3f}]")
                else:
                    logger.info(f"Report {correction}: 0 non-zero edges (all thresholded out)")

                coordinates = None
                roi_names = None
                atlas_name = None
                if self._connectivity_metadata is not None:
                    coordinates = self._connectivity_metadata.get('roi_coordinates')
                    roi_names = self._connectivity_metadata.get('roi_names')
                    atlas_name = self._connectivity_metadata.get('atlas_name')

                # Pass unthresholded matrix only for the first correction to avoid redundancy
                unthresholded_for_display = t_matrix_unthresholded if i == 0 else None

                self.report.add_connectivity_results_section(
                    t_matrix=t_matrix_thresholded,
                    p_matrix=p_matrix,
                    contrast_name=contrast_name,
                    correction=correction,
                    threshold=threshold,
                    coordinates=coordinates,
                    roi_names=roi_names,
                    atlas_name=atlas_name,
                    edge_table=edge_table,
                    df=df,
                    t_matrix_unthresholded=unthresholded_for_display,
                )

                # Add permutation null distribution plot if available
                if correction == "fwer_perm":
                    null_dist = self.connectivity_inference.null_distributions.get(contrast_name, {}).get(correction)
                    if null_dist is not None:
                        n_perm = len(null_dist)
                        alpha = self.config.get("alpha_corrected", 0.05)
                        self.report.add_permutation_null_distribution(
                            h0_distribution=null_dist,
                            alpha=alpha,
                            contrast_name=contrast_name,
                            n_perm=n_perm,
                        )
                        logger.info(f"Added null distribution plot for {contrast_name} (FWER permutation)")

        # Save report with BIDS-compatible naming
        bids_prefix = self._generate_bids_prefix()
        if len(contrast_names) == 1:
            report_filename = f"{bids_prefix}_contrast-{contrast_names[0]}_report.html"
        else:
            report_filename = f"{bids_prefix}_report.html"
        report_path = self.report.save(report_filename)
        logger.info(f"Report saved to: {report_path}")

        return report_path
