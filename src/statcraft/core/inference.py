"""
Statistical inference module for second-level analysis.

This module handles:
- Thresholding (uncorrected, FDR, FWER)
- Permutation testing
- Cluster-level inference
- Cluster table generation
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label, generate_binary_structure
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, new_img_like
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import non_parametric_inference

logger = logging.getLogger(__name__)


class StatisticalInference:
    """
    Statistical inference for second-level neuroimaging analysis.

    Handles thresholding, multiple comparison correction, and cluster
    table generation.

    Parameters
    ----------
    alpha_corrected : float
        Significance level for CORRECTED thresholds (FDR, Bonferroni,
        permutation). This controls the family-wise error rate (FWER) or
        false discovery rate (FDR). Default: 0.05.
    alpha_uncorrected : float
        Significance level for UNCORRECTED analysis. Also called
        "cluster-forming threshold" - defines which voxels are considered
        active for cluster detection. Default: 0.001.
    cluster_threshold : int
        Minimum cluster size in voxels. Default: 10.
    two_sided : bool
        Whether to use two-sided tests. Default: True.

    Notes
    -----
    Two different significance levels (alpha) are used in neuroimaging:

    - **alpha_uncorrected** (default: 0.001): Significance level for uncorrected
      analysis. This is the cluster-forming threshold that determines which
      voxels are included in the statistical map before cluster detection.

    - **alpha_corrected** (default: 0.05): Significance level for corrected
      analyses (FDR, Bonferroni, permutation). This controls the error rate
      after multiple comparison correction.

    Attributes
    ----------
    thresholded_maps : dict
        Dictionary of thresholded statistical maps.
    cluster_tables : dict
        Dictionary of cluster tables.
    threshold_values : dict
        Dictionary storing the actual z-thresholds used for each correction method.
    """

    def __init__(
        self,
        alpha_corrected: float = 0.05,
        alpha_uncorrected: float = 0.001,
        cluster_threshold: int = 10,
        two_sided: bool = True,
    ):
        self.alpha_corrected = alpha_corrected
        self.alpha_uncorrected = alpha_uncorrected
        self.cluster_threshold = cluster_threshold
        self.two_sided = two_sided

        self.thresholded_maps: Dict[str, Dict[str, nib.Nifti1Image]] = {}
        self.cluster_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.threshold_values: Dict[str, Dict[str, float]] = {}  # Store actual threshold values
        self._inference_results: Dict[str, Any] = {}
    
    def threshold_uncorrected(
        self,
        stat_map: nib.Nifti1Image,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
    ) -> Tuple[nib.Nifti1Image, pd.DataFrame]:
        """
        Apply uncorrected p-value threshold.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map in z-score scale (required by nilearn.glm.threshold_stats_img).
            Using non-z-scaled statistics will produce unreliable thresholding results.
        alpha : float, optional
            Significance level (p-value threshold). Default: self.alpha_uncorrected.
        contrast_name : str
            Name for the contrast (for storing results).

        Returns
        -------
        tuple of (nibabel.Nifti1Image, pd.DataFrame)
            Thresholded map and cluster table.
        """
        if alpha is None:
            alpha = self.alpha_uncorrected

        logger.info(f"Applying uncorrected threshold: p < {alpha}")

        # Threshold the map
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, module='nilearn')
            warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
            thresholded_map, threshold = threshold_stats_img(
                stat_map,
                alpha=alpha,
                height_control=None,  # Uncorrected
                cluster_threshold=self.cluster_threshold,
                two_sided=self.two_sided,
            )

        # Generate cluster table
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
            cluster_table = get_clusters_table(
                stat_map,
                stat_threshold=threshold,
                cluster_threshold=self.cluster_threshold,
                two_sided=self.two_sided,
            )

        # Debug: check for positive and negative clusters
        if len(cluster_table) > 0 and 'Peak Stat' in cluster_table.columns:
            n_positive = (cluster_table['Peak Stat'] > 0).sum()
            n_negative = (cluster_table['Peak Stat'] < 0).sum()
            logger.info(f"Cluster table contains {n_positive} positive and {n_negative} negative clusters")

        # Store results
        key = f"{contrast_name}_uncorrected_alpha{alpha}"
        self.thresholded_maps.setdefault(contrast_name, {})["uncorrected"] = thresholded_map
        self.cluster_tables.setdefault(contrast_name, {})["uncorrected"] = cluster_table
        self.threshold_values.setdefault(contrast_name, {})["uncorrected"] = float(threshold)

        logger.info(f"Found {len(cluster_table)} clusters at uncorrected α < {alpha} (z-threshold: {threshold:.4f})")
        
        return thresholded_map, cluster_table
    
    def threshold_fdr(
        self,
        stat_map: nib.Nifti1Image,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
    ) -> Tuple[nib.Nifti1Image, pd.DataFrame]:
        """
        Apply FDR (False Discovery Rate) correction.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map in z-score scale (required by nilearn.glm.threshold_stats_img).
            Using non-z-scaled statistics will produce unreliable thresholding results.
        alpha : float, optional
            FDR level. Default: self.alpha_corrected.
        contrast_name : str
            Name for the contrast.

        Returns
        -------
        tuple of (nibabel.Nifti1Image, pd.DataFrame)
            Thresholded map and cluster table.
        """
        if alpha is None:
            alpha = self.alpha_corrected
        
        logger.info(f"Applying FDR correction: q < {alpha}")

        try:
            # Threshold with FDR
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, module='nilearn')
                warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
                thresholded_map, threshold = threshold_stats_img(
                    stat_map,
                    alpha=alpha,
                    height_control="fdr",
                    cluster_threshold=self.cluster_threshold,
                    two_sided=self.two_sided,
                )

            # Generate cluster table
            if threshold is not None and not np.isinf(threshold):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
                    cluster_table = get_clusters_table(
                        stat_map,
                        stat_threshold=threshold,
                        cluster_threshold=self.cluster_threshold,
                        two_sided=self.two_sided,
                    )
                # Store threshold value
                self.threshold_values.setdefault(contrast_name, {})["fdr"] = float(threshold)
                logger.info(f"Found {len(cluster_table)} clusters at FDR q < {alpha} (z-threshold: {threshold:.4f})")
            else:
                cluster_table = pd.DataFrame()
                logger.warning("No voxels survive FDR correction")
                self.threshold_values.setdefault(contrast_name, {})["fdr"] = np.inf

        except Exception as e:
            logger.warning(f"FDR correction failed: {e}")
            thresholded_map = new_img_like(stat_map, np.zeros(stat_map.shape))
            cluster_table = pd.DataFrame()
            self.threshold_values.setdefault(contrast_name, {})["fdr"] = np.inf

        # Store results
        self.thresholded_maps.setdefault(contrast_name, {})["fdr"] = thresholded_map
        self.cluster_tables.setdefault(contrast_name, {})["fdr"] = cluster_table
        
        return thresholded_map, cluster_table
    
    def threshold_fwer_bonferroni(
        self,
        stat_map: nib.Nifti1Image,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
    ) -> Tuple[nib.Nifti1Image, pd.DataFrame]:
        """
        Apply Bonferroni FWER correction.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map in z-score scale (required by nilearn.glm.threshold_stats_img).
            Using non-z-scaled statistics will produce unreliable thresholding results.
        alpha : float, optional
            FWER level. Default: self.alpha_corrected.
        contrast_name : str
            Name for the contrast.

        Returns
        -------
        tuple of (nibabel.Nifti1Image, pd.DataFrame)
            Thresholded map and cluster table.
        """
        if alpha is None:
            alpha = self.alpha_corrected
        
        logger.info(f"Applying Bonferroni correction: p < {alpha}")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, module='nilearn')
                warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
                thresholded_map, threshold = threshold_stats_img(
                    stat_map,
                    alpha=alpha,
                    height_control="bonferroni",
                    cluster_threshold=self.cluster_threshold,
                    two_sided=self.two_sided,
                )

            if threshold is not None and not np.isinf(threshold):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
                    cluster_table = get_clusters_table(
                        stat_map,
                        stat_threshold=threshold,
                        cluster_threshold=self.cluster_threshold,
                        two_sided=self.two_sided,
                    )
                # Store threshold value
                self.threshold_values.setdefault(contrast_name, {})["bonferroni"] = float(threshold)
                logger.info(f"Found {len(cluster_table)} clusters at Bonferroni p < {alpha} (z-threshold: {threshold:.4f})")
            else:
                cluster_table = pd.DataFrame()
                logger.warning("No voxels survive Bonferroni correction")
                self.threshold_values.setdefault(contrast_name, {})["bonferroni"] = np.inf

        except Exception as e:
            logger.warning(f"Bonferroni correction failed: {e}")
            thresholded_map = new_img_like(stat_map, np.zeros(stat_map.shape))
            cluster_table = pd.DataFrame()
            self.threshold_values.setdefault(contrast_name, {})["bonferroni"] = np.inf

        # Store results
        self.thresholded_maps.setdefault(contrast_name, {})["bonferroni"] = thresholded_map
        self.cluster_tables.setdefault(contrast_name, {})["bonferroni"] = cluster_table
        
        return thresholded_map, cluster_table
    
    def threshold_fwer_permutation(
        self,
        second_level_input: List[Union[str, nib.Nifti1Image]],
        design_matrix: pd.DataFrame,
        contrast: Union[str, np.ndarray, List[float]],
        n_perm: int = 1000,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        smoothing_fwhm: Optional[float] = None,
    ) -> Tuple[nib.Nifti1Image, pd.DataFrame, Dict]:
        """
        Apply FWER correction using permutation testing.

        Uses max-stat distribution for voxel-level FWER control.

        Parameters
        ----------
        second_level_input : list
            List of images or paths.
        design_matrix : pd.DataFrame
            Design matrix.
        contrast : str, array, or list
            Contrast definition.
        n_perm : int
            Number of permutations. Default: 1000.
        alpha : float, optional
            FWER level. Default: self.alpha_corrected.
        contrast_name : str
            Name for the contrast.
        n_jobs : int
            Number of parallel jobs.
        random_state : int, optional
            Random state for reproducibility.
        smoothing_fwhm : float, optional
            Smoothing kernel FWHM in mm. If None, no smoothing is applied.

        Returns
        -------
        tuple of (nibabel.Nifti1Image, pd.DataFrame, dict)
            Thresholded map, cluster table, and additional results.
        """
        if alpha is None:
            alpha = self.alpha_corrected
        
        logger.info(f"Running permutation test with {n_perm} permutations")
        if smoothing_fwhm is not None:
            logger.info(f"Applying smoothing with FWHM={smoothing_fwhm} mm")
        
        # Run non-parametric inference
        try:
            perm_results = non_parametric_inference(
                second_level_input,
                design_matrix=design_matrix,
                second_level_contrast=contrast,
                n_perm=n_perm,
                two_sided_test=self.two_sided,
                threshold=self.alpha_uncorrected,
                n_jobs=n_jobs,
                random_state=random_state,
                smoothing_fwhm=smoothing_fwhm,
            )
            
            # Extract results
            # perm_results is a dict with various outputs
            if isinstance(perm_results, dict):
                logp_max_map = perm_results.get("logp_max_t", perm_results.get("logp_max_stat"))
                t_map = perm_results.get("t", perm_results.get("stat"))
                h0_max_t = perm_results.get("h0_max_t", perm_results.get("h0_max_stat"))
            else:
                # Older nilearn versions return a single map
                logp_max_map = perm_results
                t_map = None
                h0_max_t = None

            # If h0_max_t not available in perm_results, try to extract from permuted_ols directly
            if h0_max_t is None:
                logger.info("Null distribution not found in non_parametric_inference output, trying permuted_ols directly...")
                try:
                    from nilearn.mass_univariate import permuted_ols
                    from nilearn.maskers import NiftiMasker
                    from nilearn._utils.niimg_conversions import check_niimg

                    # Create masker from first image
                    first_img = check_niimg(second_level_input[0])
                    masker = NiftiMasker(mask_strategy='epi').fit(first_img)

                    # Transform images to 2D array
                    Y = np.vstack([masker.transform(img) for img in second_level_input])

                    # Prepare contrast
                    if isinstance(contrast, (list, np.ndarray)):
                        contrast_vec = np.array(contrast).reshape(-1, 1)
                    else:
                        contrast_vec = np.ones((len(design_matrix.columns), 1))

                    # Apply contrast to design matrix
                    X = design_matrix.values
                    tested_var = X @ contrast_vec

                    # Run permuted_ols with dict output to get h0 distribution
                    ols_results = permuted_ols(
                        tested_var,
                        Y,
                        confounding_vars=None,
                        model_intercept=False,
                        n_perm=n_perm,
                        two_sided_test=self.two_sided,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        verbose=0,
                        output_type='dict',
                    )
                    h0_max_t = ols_results.get("h0_max_t")
                    logger.info(f"Successfully extracted null distribution with {len(h0_max_t) if h0_max_t is not None else 0} permutations")
                except Exception as e:
                    logger.warning(f"Could not extract null distribution: {e}")
                    h0_max_t = None

            # Threshold at alpha (for logp map thresholding)
            logp_threshold = -np.log10(alpha)
            logp_data = get_data(logp_max_map)
            thresholded_data = np.where(logp_data >= logp_threshold, logp_data, 0)
            thresholded_map = new_img_like(logp_max_map, thresholded_data)

            # Calculate actual t-statistic threshold from h0 distribution
            actual_t_threshold = None
            if h0_max_t is not None:
                h0_array = np.asarray(h0_max_t).ravel()
                # For two-sided test at alpha=0.05, we want the 95th percentile
                actual_t_threshold = float(np.percentile(h0_array, (1 - alpha) * 100))
                logger.info(f"Actual t-statistic threshold from permutation distribution: {actual_t_threshold:.3f}")

            # Generate cluster table
            if np.any(thresholded_data > 0):
                cluster_table = get_clusters_table(
                    logp_max_map,
                    stat_threshold=logp_threshold,
                    cluster_threshold=self.cluster_threshold,
                )
            else:
                cluster_table = pd.DataFrame()
                logger.warning("No voxels survive permutation-based FWER correction")

            additional_results = {
                "logp_max_map": logp_max_map,
                "t_map": t_map,
                "logp_threshold": logp_threshold,
                "actual_t_threshold": actual_t_threshold,
                "n_perm": n_perm,
                "h0_max_t": h0_max_t,
                "alpha": alpha,
            }
            
        except Exception as e:
            logger.error(f"Permutation test failed: {e}")
            raise
        
        # Store results
        self.thresholded_maps.setdefault(contrast_name, {})["fwer_perm"] = thresholded_map
        self.cluster_tables.setdefault(contrast_name, {})["fwer_perm"] = cluster_table
        # Store the actual t-statistic threshold (not logp threshold)
        self.threshold_values.setdefault(contrast_name, {})["fwer_perm"] = float(actual_t_threshold) if actual_t_threshold is not None else float(logp_threshold)
        self._inference_results[f"{contrast_name}_perm"] = additional_results

        logger.info(f"Found {len(cluster_table)} clusters at permutation FWER p < {alpha}")

        return thresholded_map, cluster_table, additional_results
    
    def run_all_corrections(
        self,
        stat_map: nib.Nifti1Image,
        contrast_name: str = "contrast",
        include_permutation: bool = False,
        permutation_inputs: Optional[Dict] = None,
    ) -> Dict[str, Tuple[nib.Nifti1Image, pd.DataFrame]]:
        """
        Run all correction methods on a statistical map.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map in z-score scale (required by nilearn.glm.threshold_stats_img).
            Using non-z-scaled statistics will produce unreliable thresholding results.
        contrast_name : str
            Name for the contrast.
        include_permutation : bool
            Whether to include permutation testing.
        permutation_inputs : dict, optional
            Inputs for permutation testing (images, design_matrix, contrast).

        Returns
        -------
        dict
            Dictionary of correction method -> (thresholded_map, cluster_table).
        """
        results = {}
        
        # Uncorrected
        results["uncorrected"] = self.threshold_uncorrected(
            stat_map, contrast_name=contrast_name
        )
        
        # FDR
        results["fdr"] = self.threshold_fdr(stat_map, contrast_name=contrast_name)
        
        # Bonferroni
        results["bonferroni"] = self.threshold_fwer_bonferroni(
            stat_map, contrast_name=contrast_name
        )
        
        # Permutation (if requested)
        if include_permutation and permutation_inputs is not None:
            try:
                thresh_map, cluster_table, _ = self.threshold_fwer_permutation(
                    second_level_input=permutation_inputs["images"],
                    design_matrix=permutation_inputs["design_matrix"],
                    contrast=permutation_inputs["contrast"],
                    contrast_name=contrast_name,
                    n_perm=permutation_inputs.get("n_perm", 1000),
                    n_jobs=permutation_inputs.get("n_jobs", 1),
                )
                results["fwer_perm"] = (thresh_map, cluster_table)
            except Exception as e:
                logger.warning(f"Permutation test skipped: {e}")
        
        return results
    
    def get_cluster_table(
        self,
        contrast_name: str,
        correction: str = "uncorrected",
    ) -> pd.DataFrame:
        """
        Get a cluster table for a specific contrast and correction.
        
        Parameters
        ----------
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method: "uncorrected", "fdr", "bonferroni", "fwer_perm".
        
        Returns
        -------
        pd.DataFrame
            Cluster table.
        """
        if contrast_name not in self.cluster_tables:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        if correction not in self.cluster_tables[contrast_name]:
            raise KeyError(f"Correction '{correction}' not found for contrast '{contrast_name}'")
        
        return self.cluster_tables[contrast_name][correction]
    
    def get_thresholded_map(
        self,
        contrast_name: str,
        correction: str = "uncorrected",
    ) -> nib.Nifti1Image:
        """
        Get a thresholded map for a specific contrast and correction.
        
        Parameters
        ----------
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method.
        
        Returns
        -------
        nibabel.Nifti1Image
            Thresholded statistical map.
        """
        if contrast_name not in self.thresholded_maps:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        if correction not in self.thresholded_maps[contrast_name]:
            raise KeyError(f"Correction '{correction}' not found for contrast '{contrast_name}'")
        
        return self.thresholded_maps[contrast_name][correction]
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        Save all thresholded maps and cluster tables.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory.
        prefix : str
            Prefix for filenames.
        
        Returns
        -------
        dict
            Dictionary of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save thresholded maps
        for contrast_name, corrections in self.thresholded_maps.items():
            for correction, stat_map in corrections.items():
                if prefix:
                    filename = f"{prefix}_{contrast_name}_{correction}_thresh.nii.gz"
                else:
                    filename = f"{contrast_name}_{correction}_thresh.nii.gz"
                
                filepath = output_dir / filename
                nib.save(stat_map, filepath)
                saved_files[f"{contrast_name}_{correction}_map"] = filepath
        
        # Save cluster tables
        for contrast_name, corrections in self.cluster_tables.items():
            for correction, table in corrections.items():
                if len(table) == 0:
                    continue
                    
                if prefix:
                    filename = f"{prefix}_{contrast_name}_{correction}_clusters.tsv"
                else:
                    filename = f"{contrast_name}_{correction}_clusters.tsv"
                
                filepath = output_dir / filename
                table.to_csv(filepath, sep="\t", index=False)
                saved_files[f"{contrast_name}_{correction}_table"] = filepath
        
        logger.info(f"Saved {len(saved_files)} result files to {output_dir}")
        return saved_files
    
    def summary(self) -> str:
        """
        Get a text summary of inference results.

        Returns
        -------
        str
            Summary text.
        """
        lines = ["Statistical Inference Summary", "=" * 30, ""]
        lines.append("Significance Levels (α):")
        lines.append(f"  - Uncorrected (cluster-forming): α < {self.alpha_uncorrected}")
        lines.append(f"  - Corrected (FDR/Bonferroni/perm): α < {self.alpha_corrected}")
        lines.append(f"Cluster threshold: {self.cluster_threshold} voxels")
        lines.append(f"Two-sided: {self.two_sided}")
        lines.append("")

        for contrast_name, corrections in self.cluster_tables.items():
            lines.append(f"Contrast: {contrast_name}")
            for correction, table in corrections.items():
                n_clusters = len(table)
                lines.append(f"  {correction}: {n_clusters} clusters")
            lines.append("")

        return "\n".join(lines)

    def perform_extra_cluster_analysis(
        self,
        stat_map: nib.Nifti1Image,
        thresholded_map: nib.Nifti1Image,
        contrast_name: str,
        correction: str,
        overlap_threshold: float = 5.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform detailed cluster analysis including Harvard-Oxford probability overlap
        and cortical/non-cortical statistics.

        Parameters
        ----------
        stat_map : Nifti1Image
            Statistical map (e.g., z-score map).
        thresholded_map : Nifti1Image
            Thresholded statistical map.
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method name.
        overlap_threshold : float
            Minimum percentage threshold for reporting atlas region overlaps (default: 5.0).

        Returns
        -------
        enhanced_cluster_table : pd.DataFrame
            Enhanced cluster table with region percentages.
        cortical_summary : pd.DataFrame
            Summary statistics for cortical/non-cortical positive/negative clusters.
        """
        from nilearn import datasets
        from nilearn.image import resample_to_img, math_img

        logger.info(f"Performing extra cluster analysis for {contrast_name}/{correction}")

        # Get the appropriate threshold for this correction method
        # Use the stored threshold value that was computed during inference
        if contrast_name in self.threshold_values and correction in self.threshold_values[contrast_name]:
            stat_threshold = self.threshold_values[contrast_name][correction]
            logger.info(f"Using {correction} threshold: {stat_threshold:.4f}")
        else:
            # Fallback to uncorrected threshold if not found
            stat_threshold = self.alpha_uncorrected
            logger.warning(f"Could not find threshold for {contrast_name}/{correction}, using uncorrected α: {stat_threshold:.4f}")

        # Get cluster table with label maps using the correction-specific threshold
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')
            cluster_table_result = get_clusters_table(
                stat_map,
                stat_threshold=stat_threshold,
                cluster_threshold=self.cluster_threshold,
                two_sided=self.two_sided,
                return_label_maps=True
            )

        # Extract label maps - handle different return formats
        if isinstance(cluster_table_result, tuple):
            if len(cluster_table_result) == 3:
                # (table, pos_labels, neg_labels) - two-sided test
                table_df, label_maps_pos, label_maps_neg = cluster_table_result
            elif len(cluster_table_result) == 2:
                # (table, label_map) - one-sided test or no negative clusters
                table_df, label_maps_pos = cluster_table_result
                label_maps_neg = None
            else:
                logger.warning(f"Unexpected tuple length from get_clusters_table: {len(cluster_table_result)}")
                table_df = cluster_table_result[0]
                label_maps_pos = None
                label_maps_neg = None
        else:
            # DataFrame only (old API or no return_label_maps support)
            table_df = cluster_table_result
            label_maps_pos = None
            label_maps_neg = None

        if table_df.empty:
            logger.warning(f"No clusters found for {contrast_name}/{correction}")
            return pd.DataFrame(), pd.DataFrame()

        # Load Harvard-Oxford probabilistic atlas
        logger.info("Loading Harvard-Oxford cortical and subcortical atlases")
        ho_cortical = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
        ho_subcortical = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')

        # Log atlas file paths for verification
        logger.info(f"Harvard-Oxford cortical atlas file: {ho_cortical['maps']}")
        logger.info(f"Harvard-Oxford subcortical atlas file: {ho_subcortical['maps']}")

        # Load tissue probability maps (for cortical/non-cortical classification)
        logger.info("Loading tissue probability maps")
        try:
            from nilearn.datasets import load_mni152_gm_mask, load_mni152_wm_mask
            gm_prob_map = load_mni152_gm_mask()
            wm_prob_map = load_mni152_wm_mask()
        except:
            # Fallback: use simple thresholding approach
            logger.warning("Could not load MNI tissue probability maps, using simplified cortical/non-cortical classification")
            gm_prob_map = None
            wm_prob_map = None

        # Resample atlas and tissue maps to stat_map space
        logger.info("Resampling atlases to stat map space")

        # Process each atlas region
        cortical_maps = []
        cortical_labels = ho_cortical['labels'][1:]  # Skip background

        # Load atlas maps (handle both file paths and Nifti1Image objects)
        if isinstance(ho_cortical['maps'], str):
            cortical_atlas = nib.load(ho_cortical['maps'])
        else:
            cortical_atlas = ho_cortical['maps']

        cortical_prob_data = get_data(cortical_atlas)
        for i in range(len(cortical_labels)):
            region_data = cortical_prob_data[..., i]
            region_img = new_img_like(cortical_atlas, region_data)
            cortical_maps.append(resample_to_img(
                region_img, stat_map,
                interpolation='continuous',
                copy_header=True,
                force_resample=True
            ))

        subcortical_maps = []
        subcortical_labels = ho_subcortical['labels'][1:]  # Skip background

        # Load atlas maps (handle both file paths and Nifti1Image objects)
        if isinstance(ho_subcortical['maps'], str):
            subcortical_atlas = nib.load(ho_subcortical['maps'])
        else:
            subcortical_atlas = ho_subcortical['maps']

        subcortical_prob_data = get_data(subcortical_atlas)
        for i in range(len(subcortical_labels)):
            region_data = subcortical_prob_data[..., i]
            region_img = new_img_like(subcortical_atlas, region_data)
            subcortical_maps.append(resample_to_img(
                region_img, stat_map,
                interpolation='continuous',
                copy_header=True,
                force_resample=True
            ))

        # Combine all labels and maps
        all_labels = list(cortical_labels) + list(subcortical_labels)
        all_maps = cortical_maps + subcortical_maps

        # Get stat map data
        stat_data = get_data(stat_map)

        # Resample tissue probability maps ONCE (outside the cluster loop)
        gm_data = None
        wm_data = None
        if gm_prob_map is not None and wm_prob_map is not None:
            logger.info("Resampling tissue probability maps to stat map space")
            # Use nearest neighbor interpolation for binary/mask images to avoid warnings
            gm_resampled = resample_to_img(
                gm_prob_map, stat_map,
                interpolation='nearest',
                copy_header=True,
                force_resample=True
            )
            wm_resampled = resample_to_img(
                wm_prob_map, stat_map,
                interpolation='nearest',
                copy_header=True,
                force_resample=True
            )
            gm_data = get_data(gm_resampled)
            wm_data = get_data(wm_resampled)

            logger.info(f"Initial GM data shape: {gm_data.shape}, WM data shape: {wm_data.shape}")
            logger.info(f"Stat data shape: {stat_data.shape}")

            # Ensure 3D data (squeeze out singleton dimensions if needed)
            if gm_data.ndim > 3:
                logger.info(f"Squeezing GM data from {gm_data.shape}")
                gm_data = np.squeeze(gm_data)
                logger.info(f"After squeeze: {gm_data.shape}")
            if wm_data.ndim > 3:
                logger.info(f"Squeezing WM data from {wm_data.shape}")
                wm_data = np.squeeze(wm_data)
                logger.info(f"After squeeze: {wm_data.shape}")

        # Initialize enhanced cluster information
        cluster_regions = []
        cortical_pos_stats = []
        cortical_neg_stats = []
        non_cortical_pos_stats = []
        non_cortical_neg_stats = []

        # Process each cluster
        for idx, row in table_df.iterrows():
            cluster_id = row.get('Cluster ID', idx + 1)
            peak_x, peak_y, peak_z = row['X'], row['Y'], row['Z']
            peak_stat = row.get('Peak Stat', row.get('Z', 0))
            cluster_size_mm3 = row.get('Cluster Size (mm3)', 0)

            # Determine if positive or negative cluster
            is_positive = peak_stat > 0

            # Get cluster mask from label map
            # For two-sided tests with return_label_maps=True, nilearn may return:
            # - A 2-tuple: (table, label_map_4d) where label_map_4d[...,0]=pos, label_map_4d[...,1]=neg
            # - A 3-tuple: (table, pos_labels, neg_labels) where each is 3D
            if label_maps_pos is not None:
                label_data = get_data(label_maps_pos)

                # Handle 4D label maps (last dimension separates pos/neg clusters)
                if label_data.ndim == 4:
                    if is_positive:
                        # Extract positive clusters (first slice)
                        label_data = label_data[..., 0]
                    else:
                        # Extract negative clusters (second slice)
                        if label_data.shape[-1] > 1:
                            label_data = label_data[..., 1]
                        else:
                            logger.warning(f"Expected 4D label map to have negative clusters, but shape is {label_data.shape}")
                            continue
            elif not is_positive and label_maps_neg is not None:
                # Separate negative label map provided
                label_data = get_data(label_maps_neg)
            else:
                logger.warning(f"Could not get label map for cluster {cluster_id}")
                continue

            # Find cluster mask (label maps use cluster IDs)
            cluster_mask = (label_data == cluster_id)

            if not np.any(cluster_mask):
                logger.warning(f"Empty cluster mask for cluster {cluster_id}")
                continue

            # Calculate region overlaps
            cluster_volume = np.sum(cluster_mask)
            region_percentages = {}

            for region_name, region_map in zip(all_labels, all_maps):
                region_data = get_data(region_map)

                # Calculate overlap as sum of probabilities in cluster region
                overlap = np.sum(region_data[cluster_mask])

                # Debug: check if values are already percentages (0-100) or fractions (0-1)
                if idx == table_df.index[0] and region_name == all_labels[0]:
                    sample_values = region_data[cluster_mask][region_data[cluster_mask] > 0]
                    if len(sample_values) > 0:
                        logger.info(f"Atlas value range check: min={sample_values.min():.2f}, max={sample_values.max():.2f}, mean={sample_values.mean():.2f}")

                # Normalize by cluster size to get average probability
                # Harvard-Oxford atlas stores probabilities as 0-100 (percentages), not 0-1 (fractions)
                if cluster_volume > 0:
                    avg_prob = overlap / cluster_volume  # Already in percentage scale
                    if avg_prob >= overlap_threshold:
                        region_percentages[region_name] = avg_prob

            # Sort regions by percentage (descending)
            sorted_regions = sorted(region_percentages.items(), key=lambda x: x[1], reverse=True)
            region_str = ", ".join([f"{name} ({pct:.1f}%)" for name, pct in sorted_regions])

            cluster_regions.append({
                'Cluster ID': cluster_id,
                'Peak Stat': peak_stat,
                'X': peak_x,
                'Y': peak_y,
                'Z': peak_z,
                'Cluster Size (mm3)': cluster_size_mm3,
                'Regions (% overlap)': region_str if region_str else "No significant overlap"
            })

            # Classify peak as cortical or non-cortical using tissue probability maps
            if gm_data is not None and wm_data is not None:
                # Convert MNI coordinates to voxel indices
                affine_inv = np.linalg.inv(stat_map.affine)
                peak_vox = affine_inv.dot([peak_x, peak_y, peak_z, 1])[:3].astype(int)

                # Get tissue probabilities at peak
                try:
                    gm_prob_at_peak = gm_data[peak_vox[0], peak_vox[1], peak_vox[2]]
                    wm_prob_at_peak = wm_data[peak_vox[0], peak_vox[1], peak_vox[2]]
                    is_cortical = gm_prob_at_peak > wm_prob_at_peak
                except (IndexError, ValueError) as e:
                    # Peak outside volume or other indexing issue, use cluster centroid
                    logger.warning(f"Could not get tissue probability at peak for cluster {cluster_id}: {e}. Using cluster majority.")
                    logger.info(f"DEBUG: gm_data.shape={gm_data.shape}, wm_data.shape={wm_data.shape}, cluster_mask.shape={cluster_mask.shape}")
                    logger.info(f"DEBUG: cluster_mask dtype={cluster_mask.dtype}, cluster_mask sum={np.sum(cluster_mask)}")
                    try:
                        gm_overlap = np.sum(gm_data[cluster_mask])
                        wm_overlap = np.sum(wm_data[cluster_mask])
                        is_cortical = gm_overlap > wm_overlap
                    except Exception as e2:
                        logger.error(f"Failed during cluster majority calculation: {e2}")
                        logger.error(f"This suggests shape mismatch: gm_data.shape={gm_data.shape}, cluster_mask.shape={cluster_mask.shape}")
                        # Fallback to cortical heuristic
                        is_cortical = any('cortex' in region.lower() or 'cortical' in region.lower()
                                   for region in region_percentages.keys())
            else:
                # Simplified: assume cortical regions are cortical, others non-cortical
                # This is a rough approximation
                is_cortical = any('cortex' in region.lower() or 'cortical' in region.lower()
                           for region in region_percentages.keys())

            # Calculate average stat in cluster
            avg_stat = np.mean(np.abs(stat_data[cluster_mask]))

            # Categorize cluster
            if is_cortical and is_positive:
                cortical_pos_stats.append({'avg_stat': avg_stat, 'size_mm3': cluster_size_mm3})
            elif is_cortical and not is_positive:
                cortical_neg_stats.append({'avg_stat': avg_stat, 'size_mm3': cluster_size_mm3})
            elif not is_cortical and is_positive:
                non_cortical_pos_stats.append({'avg_stat': avg_stat, 'size_mm3': cluster_size_mm3})
            else:  # Non-cortical and negative
                non_cortical_neg_stats.append({'avg_stat': avg_stat, 'size_mm3': cluster_size_mm3})

        # Create enhanced cluster table
        enhanced_table = pd.DataFrame(cluster_regions)

        # Create cortical/non-cortical summary table
        def summarize_stats(stats_list, category):
            if not stats_list:
                return {
                    'Category': category,
                    'N Clusters': 0,
                    'Avg |Stat|': np.nan,
                    'Avg Size (mm3)': np.nan,
                    'Total Size (mm3)': 0
                }
            return {
                'Category': category,
                'N Clusters': len(stats_list),
                'Avg |Stat|': np.mean([s['avg_stat'] for s in stats_list]),
                'Avg Size (mm3)': np.mean([s['size_mm3'] for s in stats_list]),
                'Total Size (mm3)': np.sum([s['size_mm3'] for s in stats_list])
            }

        cortical_summary = pd.DataFrame([
            summarize_stats(cortical_pos_stats, 'Cortical (positive)'),
            summarize_stats(cortical_neg_stats, 'Cortical (negative)'),
            summarize_stats(non_cortical_pos_stats, 'Non-Cortical (positive)'),
            summarize_stats(non_cortical_neg_stats, 'Non-Cortical (negative)'),
        ])

        logger.info(f"Extra cluster analysis complete: {len(enhanced_table)} clusters analyzed")

        return enhanced_table, cortical_summary


class ConnectivityInference:
    """
    Statistical inference for connectivity matrix analysis.
    
    Handles edge-wise thresholding and multiple comparison correction
    for connectivity matrices. Does NOT perform cluster analysis since
    edges are not spatially contiguous.
    
    Parameters
    ----------
    alpha_corrected : float
        Significance level for corrected thresholds (FDR, Bonferroni).
        Default: 0.05.
    alpha_uncorrected : float
        Significance level for uncorrected analysis. Default: 0.001.
    two_sided : bool
        Whether to use two-sided tests. Default: True.
    """
    
    def __init__(
        self,
        alpha_corrected: float = 0.05,
        alpha_uncorrected: float = 0.001,
        two_sided: bool = True,
    ):
        self.alpha_corrected = alpha_corrected
        self.alpha_uncorrected = alpha_uncorrected
        self.two_sided = two_sided
        
        self.thresholded_matrices: Dict[str, Dict[str, np.ndarray]] = {}
        self.threshold_values: Dict[str, Dict[str, float]] = {}
        self.edge_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.null_distributions: Dict[str, Dict[str, np.ndarray]] = {}  # Store null distributions from permutation
    
    def threshold_uncorrected(
        self,
        t_matrix: np.ndarray,
        p_matrix: np.ndarray,
        df: int,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply uncorrected p-value threshold to connectivity matrix.
        
        Parameters
        ----------
        t_matrix : np.ndarray
            T-statistic matrix (n_rois x n_rois).
        p_matrix : np.ndarray
            P-value matrix (n_rois x n_rois).
        df : int
            Degrees of freedom.
        alpha : float, optional
            Significance level. Default: self.alpha_uncorrected.
        contrast_name : str
            Name for the contrast.
        
        Returns
        -------
        tuple of (np.ndarray, pd.DataFrame)
            Thresholded t-matrix and significant edge table.
        """
        if alpha is None:
            alpha = self.alpha_uncorrected
        
        logger.info(f"Applying uncorrected threshold: p < {alpha}")
        
        # Get t-threshold from alpha
        if self.two_sided:
            t_threshold = stats.t.ppf(1 - alpha/2, df)
        else:
            t_threshold = stats.t.ppf(1 - alpha, df)
        
        # Create mask of significant edges
        sig_mask = np.abs(t_matrix) > t_threshold
        
        # Create thresholded matrix
        thresholded = t_matrix.copy()
        thresholded[~sig_mask] = 0
        
        # Create edge table
        edge_table = self._create_edge_table(t_matrix, p_matrix, sig_mask)
        
        # Store results
        self.thresholded_matrices.setdefault(contrast_name, {})["uncorrected"] = thresholded
        self.threshold_values.setdefault(contrast_name, {})["uncorrected"] = float(t_threshold)
        self.edge_tables.setdefault(contrast_name, {})["uncorrected"] = edge_table
        
        n_sig = np.sum(sig_mask) // 2  # Divide by 2 for symmetric matrix
        logger.info(f"Found {n_sig} significant edges at uncorrected α < {alpha} (t-threshold: {t_threshold:.4f})")
        
        return thresholded, edge_table
    
    def threshold_fdr(
        self,
        t_matrix: np.ndarray,
        p_matrix: np.ndarray,
        df: int,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply FDR correction to connectivity matrix.
        
        Parameters
        ----------
        t_matrix : np.ndarray
            T-statistic matrix (n_rois x n_rois).
        p_matrix : np.ndarray
            P-value matrix (n_rois x n_rois).
        df : int
            Degrees of freedom.
        alpha : float, optional
            FDR level. Default: self.alpha_corrected.
        contrast_name : str
            Name for the contrast.
        
        Returns
        -------
        tuple of (np.ndarray, pd.DataFrame)
            Thresholded t-matrix and significant edge table.
        """
        if alpha is None:
            alpha = self.alpha_corrected
        
        logger.info(f"Applying FDR correction: q < {alpha}")
        
        # Get upper triangle p-values (excluding diagonal)
        triu_indices = np.triu_indices(p_matrix.shape[0], k=1)
        p_vals = p_matrix[triu_indices]
        
        # Apply FDR correction (Benjamini-Hochberg)
        n_edges = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        sorted_pvals = p_vals[sorted_idx]
        
        # BH procedure: find largest k where p(k) <= k/m * q
        ranks = np.arange(1, n_edges + 1)
        threshold_pvals = ranks / n_edges * alpha
        
        # Find significant edges
        below_threshold = sorted_pvals <= threshold_pvals
        if np.any(below_threshold):
            # Find the largest k where p(k) <= threshold
            max_k = np.max(np.where(below_threshold)[0])
            fdr_threshold = sorted_pvals[max_k]
            sig_edges_sorted = np.zeros(n_edges, dtype=bool)
            sig_edges_sorted[:max_k + 1] = True
            
            # Unsort to get original indices
            sig_edges = np.zeros(n_edges, dtype=bool)
            sig_edges[sorted_idx[sig_edges_sorted]] = True
        else:
            fdr_threshold = 0.0
            sig_edges = np.zeros(n_edges, dtype=bool)
        
        # Create significance mask for full matrix
        sig_mask = np.zeros_like(p_matrix, dtype=bool)
        sig_mask[triu_indices] = sig_edges
        sig_mask = sig_mask | sig_mask.T  # Make symmetric
        
        # Create thresholded matrix
        thresholded = t_matrix.copy()
        thresholded[~sig_mask] = 0
        
        # Get t-threshold corresponding to FDR p-threshold
        if np.any(sig_edges):
            t_threshold = np.min(np.abs(t_matrix[sig_mask]))
        else:
            t_threshold = np.inf
        
        # Create edge table
        edge_table = self._create_edge_table(t_matrix, p_matrix, sig_mask)
        
        # Store results
        self.thresholded_matrices.setdefault(contrast_name, {})["fdr"] = thresholded
        self.threshold_values.setdefault(contrast_name, {})["fdr"] = float(t_threshold)
        self.edge_tables.setdefault(contrast_name, {})["fdr"] = edge_table
        
        n_sig = np.sum(sig_edges)
        logger.info(f"Found {n_sig} significant edges at FDR q < {alpha}")
        
        return thresholded, edge_table
    
    def threshold_bonferroni(
        self,
        t_matrix: np.ndarray,
        p_matrix: np.ndarray,
        df: int,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply Bonferroni correction to connectivity matrix.
        
        Parameters
        ----------
        t_matrix : np.ndarray
            T-statistic matrix (n_rois x n_rois).
        p_matrix : np.ndarray
            P-value matrix (n_rois x n_rois).
        df : int
            Degrees of freedom.
        alpha : float, optional
            FWER level. Default: self.alpha_corrected.
        contrast_name : str
            Name for the contrast.
        
        Returns
        -------
        tuple of (np.ndarray, pd.DataFrame)
            Thresholded t-matrix and significant edge table.
        """
        if alpha is None:
            alpha = self.alpha_corrected
        
        logger.info(f"Applying Bonferroni correction: p < {alpha}")
        
        # Get number of edges (upper triangle excluding diagonal)
        n_rois = p_matrix.shape[0]
        n_edges = n_rois * (n_rois - 1) // 2
        
        # Bonferroni-corrected p-threshold
        p_threshold = alpha / n_edges
        
        # Get t-threshold
        if self.two_sided:
            t_threshold = stats.t.ppf(1 - p_threshold/2, df)
        else:
            t_threshold = stats.t.ppf(1 - p_threshold, df)
        
        # Create mask of significant edges
        sig_mask = np.abs(t_matrix) > t_threshold
        
        # Create thresholded matrix
        thresholded = t_matrix.copy()
        thresholded[~sig_mask] = 0
        
        # Create edge table
        edge_table = self._create_edge_table(t_matrix, p_matrix, sig_mask)
        
        # Store results
        self.thresholded_matrices.setdefault(contrast_name, {})["bonferroni"] = thresholded
        self.threshold_values.setdefault(contrast_name, {})["bonferroni"] = float(t_threshold)
        self.edge_tables.setdefault(contrast_name, {})["bonferroni"] = edge_table
        
        n_sig = np.sum(sig_mask) // 2
        logger.info(f"Found {n_sig} significant edges at Bonferroni p < {alpha} (t-threshold: {t_threshold:.4f})")
        
        return thresholded, edge_table
    
    def threshold_fwer_permutation(
        self,
        t_matrix: np.ndarray,
        p_matrix: np.ndarray,
        df: int,
        connectivity_data: np.ndarray,
        design_matrix: np.ndarray,
        contrast_vector: np.ndarray,
        n_perm: int = 1000,
        alpha: Optional[float] = None,
        contrast_name: str = "contrast",
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply FWER correction via permutation testing to connectivity matrix.
        
        Uses nilearn's permuted_ols function to properly estimate the null distribution
        of max t-statistics and determine the threshold that controls FWER at the
        specified alpha level. This implements the Freedman-Lane permutation scheme
        for proper FWER control.
        
        Parameters
        ----------
        t_matrix : np.ndarray
            T-statistic matrix (n_rois x n_rois) from the original analysis.
        p_matrix : np.ndarray
            P-value matrix (n_rois x n_rois) from the original analysis.
        df : int
            Degrees of freedom.
        connectivity_data : np.ndarray
            Vectorized connectivity data (n_subjects x n_edges). Each row is one
            subject's upper-triangle connectivity values.
        design_matrix : np.ndarray
            Design matrix (n_subjects x n_regressors).
        contrast_vector : np.ndarray
            Contrast vector for the tested effect.
        n_perm : int
            Number of permutations. Default: 1000.
        alpha : float, optional
            FWER level. Default: self.alpha_corrected.
        contrast_name : str
            Name for the contrast.
        n_jobs : int
            Number of parallel jobs for permutation testing. Default: 1.
        random_state : int, optional
            Random state for reproducibility.
        
        Returns
        -------
        tuple of (np.ndarray, pd.DataFrame)
            Thresholded t-matrix and significant edge table.
        
        Notes
        -----
        This function uses nilearn.mass_univariate.permuted_ols which implements
        the Freedman-Lane permutation scheme (Freedman & Lane, 1983). This scheme
        has been shown to provide proper FWER control for neuroimaging applications
        (Winkler et al., 2014).
        
        References
        ----------
        - Anderson & Robinson (2001). Permutation tests for linear models.
        - Winkler et al. (2014). Permutation inference for the general linear model.
        - Freedman & Lane (1983). A nonstochastic interpretation of reported
          significance levels.
        """
        from nilearn.mass_univariate import permuted_ols
        
        if alpha is None:
            alpha = self.alpha_corrected
        
        logger.info(f"Running permutation testing (FWER) using nilearn.mass_univariate.permuted_ols")
        logger.info(f"  n_perm={n_perm}, α={alpha}, n_jobs={n_jobs}")
        
        n_rois = t_matrix.shape[0]
        triu_indices = np.triu_indices(n_rois, k=1)
        n_edges = len(triu_indices[0])
        
        # Validate connectivity_data shape
        if connectivity_data.ndim != 2:
            raise ValueError(f"connectivity_data must be 2D (n_subjects x n_edges), got {connectivity_data.ndim}D")
        
        n_subjects, n_edges_data = connectivity_data.shape
        if n_edges_data != n_edges:
            raise ValueError(
                f"connectivity_data has {n_edges_data} edges but t_matrix expects {n_edges} edges"
            )
        
        logger.info(f"  Data shape: {n_subjects} subjects x {n_edges} edges")
        logger.info(f"  Design matrix shape: {design_matrix.shape}")
        logger.info(f"  Contrast vector: {contrast_vector}")
        
        # Prepare tested_vars: the variable(s) being tested
        contrast_vector = np.asarray(contrast_vector).ravel()
        if len(contrast_vector) != design_matrix.shape[1]:
            raise ValueError(
                f"Contrast vector length ({len(contrast_vector)}) doesn't match "
                f"design matrix columns ({design_matrix.shape[1]})"
            )
        
        # For permuted_ols, we need to split the design matrix into:
        # - tested_vars: columns of the design matrix corresponding to non-zero contrast weights
        # - confounding_vars: columns of the design matrix corresponding to zero contrast weights
        # 
        # This handles contrasts properly: for a contrast [1, 0, 0], column 0 is tested
        # and columns 1, 2 are confounders.
        #
        # For differential contrasts like [1, -1], we need to create a single tested variable
        # that is the linear combination specified by the contrast.
        
        contrast_nonzero = contrast_vector != 0
        nonzero_count = np.sum(contrast_nonzero)
        
        # Check if the design matrix already has an intercept (column of all 1s)
        # If so, we'll handle it explicitly
        has_intercept_col = False
        intercept_col_idx = None
        for col_idx in range(design_matrix.shape[1]):
            if np.allclose(design_matrix[:, col_idx], 1.0):
                has_intercept_col = True
                intercept_col_idx = col_idx
                break
        
        if nonzero_count == 1:
            # Simple contrast: test a single column
            tested_col_idx = np.where(contrast_nonzero)[0][0]
            tested_vars = design_matrix[:, tested_col_idx:tested_col_idx+1]  # Keep 2D
            
            # Confounders are all other columns (excluding the tested column)
            # Note: if intercept is among confounders, permuted_ols will handle it with model_intercept=True
            zero_contrast_mask = ~contrast_nonzero
            if np.sum(zero_contrast_mask) > 0:
                # Check if intercept is among the confounders
                if has_intercept_col and intercept_col_idx is not None and not contrast_nonzero[intercept_col_idx]:
                    # Remove intercept from confounders, set model_intercept=True
                    confounder_mask = zero_contrast_mask.copy()
                    confounder_mask[intercept_col_idx] = False
                    if np.sum(confounder_mask) > 0:
                        confounding_vars = design_matrix[:, confounder_mask]
                    else:
                        confounding_vars = None
                    model_intercept = True
                else:
                    confounding_vars = design_matrix[:, zero_contrast_mask]
                    model_intercept = False
            else:
                confounding_vars = None
                model_intercept = has_intercept_col  # Add intercept if design had one
                
            # Scale tested_vars by the contrast weight
            contrast_weight = contrast_vector[tested_col_idx]
            tested_vars = tested_vars * contrast_weight
            
        else:
            # Complex contrast (e.g., [1, -1] for differential): 
            # Create a single tested variable as the linear combination
            tested_vars = (design_matrix @ contrast_vector).reshape(-1, 1)
            
            # For complex contrasts, we can't easily separate confounders
            # Use residualization approach: no explicit confounders
            confounding_vars = None
            model_intercept = True  # Usually want an intercept for the mean

        
        # Log what we're testing
        if nonzero_count == 1:
            tested_col_idx = np.where(contrast_nonzero)[0][0]
            logger.info(f"  Testing column {tested_col_idx} (contrast weight: {contrast_vector[tested_col_idx]})")
            if confounding_vars is not None:
                logger.info(f"  Confounding variables: {confounding_vars.shape[1]} columns")
            logger.info(f"  Model intercept: {model_intercept}")
        else:
            logger.info(f"  Testing linear combination of {nonzero_count} columns")
            logger.info(f"  Model intercept: {model_intercept}")

        
        # Target variables: the connectivity data (edges)
        # permuted_ols expects shape (n_samples, n_descriptors) = (n_subjects, n_edges)
        target_vars = connectivity_data
        
        logger.info(f"  Running {n_perm} permutations...")
        print(f"\n>>> Performing permutation testing ({n_perm} permutations). This may take a while...")
        
        # Run permuted_ols with dict output to get null distribution
        try:
            perm_results = permuted_ols(
                tested_vars=tested_vars,
                target_vars=target_vars,
                confounding_vars=confounding_vars,
                model_intercept=model_intercept,
                n_perm=n_perm,
                two_sided_test=self.two_sided,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0,  # suppress nilearn's verbose output
                output_type='dict',
            )
        except Exception as e:
            logger.error(f"permuted_ols failed: {e}")
            raise
        
        # Extract results from permuted_ols output
        # t-statistics from original data: shape (1, n_edges) -> squeeze to (n_edges,)
        t_orig = perm_results['t'].squeeze()
        
        # Null distribution of max t-statistics: shape (1, n_perm) -> squeeze to (n_perm,)
        h0_max_t = perm_results['h0_max_t'].squeeze()
        
        # Log-p values (FWER corrected): shape (1, n_edges) -> squeeze to (n_edges,)
        logp_max_t = perm_results['logp_max_t'].squeeze()
        
        logger.info(f"  Permutation testing complete")
        logger.info(f"  Null distribution range: [{h0_max_t.min():.3f}, {h0_max_t.max():.3f}]")
        
        # Determine FWER threshold from null distribution
        # For two-sided test, threshold is the (1-alpha) percentile of max |t|
        t_threshold_fwer = float(np.percentile(h0_max_t, (1 - alpha) * 100))
        logger.info(f"  FWER threshold (α={alpha}): t ≥ {t_threshold_fwer:.4f}")
        
        # Create significance mask based on the corrected p-values
        # logp_max_t contains -log10(p), so significant if logp > -log10(alpha)
        logp_threshold = -np.log10(alpha)
        sig_edges = logp_max_t >= logp_threshold
        
        # Reconstruct full matrix from edge results
        sig_mask = np.zeros((n_rois, n_rois), dtype=bool)
        sig_mask[triu_indices] = sig_edges
        sig_mask = sig_mask | sig_mask.T  # Make symmetric
        
        # Create thresholded matrix
        thresholded = t_matrix.copy()
        thresholded[~sig_mask] = 0
        
        # Create edge table
        edge_table = self._create_edge_table(t_matrix, p_matrix, sig_mask)
        
        # Store results
        self.thresholded_matrices.setdefault(contrast_name, {})["fwer_perm"] = thresholded
        self.threshold_values.setdefault(contrast_name, {})["fwer_perm"] = t_threshold_fwer
        self.edge_tables.setdefault(contrast_name, {})["fwer_perm"] = edge_table
        self.null_distributions.setdefault(contrast_name, {})["fwer_perm"] = h0_max_t
        
        n_sig = np.sum(sig_edges)
        logger.info(f"  Found {n_sig} significant edges at FWER-corrected threshold")
        
        return thresholded, edge_table
    
    def _create_edge_table(
        self,
        t_matrix: np.ndarray,
        p_matrix: np.ndarray,
        sig_mask: np.ndarray,
        roi_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create a table of significant edges.
        
        Parameters
        ----------
        t_matrix : np.ndarray
            T-statistic matrix.
        p_matrix : np.ndarray
            P-value matrix.
        sig_mask : np.ndarray
            Boolean mask of significant edges.
        roi_names : list of str, optional
            Names of ROIs for labeling.
        
        Returns
        -------
        pd.DataFrame
            Table of significant edges with ROI pairs, t-statistics, and p-values.
        """
        n_rois = t_matrix.shape[0]
        
        if roi_names is None:
            roi_names = [f"ROI_{i+1}" for i in range(n_rois)]
        
        # Get upper triangle indices
        triu_indices = np.triu_indices(n_rois, k=1)
        
        edges = []
        for i, j in zip(triu_indices[0], triu_indices[1]):
            if sig_mask[i, j]:
                edges.append({
                    'ROI_1': roi_names[i],
                    'ROI_2': roi_names[j],
                    'ROI_1_idx': i,
                    'ROI_2_idx': j,
                    't_stat': t_matrix[i, j],
                    'p_value': p_matrix[i, j],
                    'direction': 'positive' if t_matrix[i, j] > 0 else 'negative',
                })
        
        df = pd.DataFrame(edges)
        if len(df) > 0:
            df = df.sort_values('t_stat', key=abs, ascending=False)
        
        return df
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        bids_prefix: str = "",
    ) -> Dict[str, Path]:
        """
        Save all results to disk with BIDS-compatible naming.
        
        Output structure:
        - output_dir/
            - data/           # Thresholded connectivity matrices (.npy)
            - tables/         # Edge tables (.tsv)
            - null_dist/      # Null distributions from permutation (.npy)
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory (should be root; data/, tables/, null_dist/ will be created inside).
        bids_prefix : str
            BIDS prefix for output filenames (e.g., 'sub-001_task-rest').
        
        Returns
        -------
        dict
            Dictionary of saved file paths.
        """
        output_dir = Path(output_dir)
        data_dir = output_dir / "data"
        tables_dir = output_dir / "tables"
        null_dist_dir = output_dir / "null_dist"
        
        data_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if there are any non-empty null distributions
        has_null_dists = any(
            null_dist is not None and len(null_dist) > 0
            for corrections in self.null_distributions.values()
            for null_dist in corrections.values()
        )
        if has_null_dists:
            null_dist_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save thresholded matrices
        for contrast_name, corrections in self.thresholded_matrices.items():
            for correction, matrix in corrections.items():
                # Create BIDS-compatible filename
                if bids_prefix:
                    filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_threshold.npy"
                else:
                    filename = f"contrast-{contrast_name}_stat-{correction}_threshold.npy"
                
                filepath = data_dir / filename
                np.save(filepath, matrix)
                saved_files[f"{contrast_name}_{correction}_matrix"] = filepath
                logger.debug(f"Saved thresholded matrix: {filepath}")
        
        # Save edge tables
        for contrast_name, corrections in self.edge_tables.items():
            for correction, table in corrections.items():
                if len(table) == 0:
                    logger.debug(f"Skipping empty edge table for {contrast_name}/{correction}")
                    continue
                
                # Create BIDS-compatible filename
                if bids_prefix:
                    filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_edges.tsv"
                else:
                    filename = f"contrast-{contrast_name}_stat-{correction}_edges.tsv"
                
                filepath = tables_dir / filename
                table.to_csv(filepath, sep="\t", index=False)
                saved_files[f"{contrast_name}_{correction}_edges"] = filepath
                logger.info(f"Saved edge table ({len(table)} edges): {filepath}")
        
        # Save null distributions from permutation testing
        for contrast_name, corrections in self.null_distributions.items():
            for correction, null_dist in corrections.items():
                if null_dist is None or len(null_dist) == 0:
                    logger.debug(f"Skipping empty null distribution for {contrast_name}/{correction}")
                    continue
                
                # Create BIDS-compatible filename
                if bids_prefix:
                    filename = f"{bids_prefix}_contrast-{contrast_name}_stat-{correction}_nulldist.npy"
                else:
                    filename = f"contrast-{contrast_name}_stat-{correction}_nulldist.npy"
                
                filepath = null_dist_dir / filename
                np.save(filepath, null_dist)
                saved_files[f"{contrast_name}_{correction}_nulldist"] = filepath
                logger.info(f"Saved null distribution ({len(null_dist)} permutations): {filepath}")
        
        logger.info(f"Saved {len(saved_files)} connectivity inference result files")
        return saved_files
    
    def summary(self) -> str:
        """
        Get a text summary of inference results.
        
        Returns
        -------
        str
            Summary text.
        """
        lines = ["Connectivity Inference Summary", "=" * 30, ""]
        lines.append("Significance Levels (α):")
        lines.append(f"  - Uncorrected: α < {self.alpha_uncorrected}")
        lines.append(f"  - Corrected (FDR/Bonferroni): α < {self.alpha_corrected}")
        lines.append(f"Two-sided: {self.two_sided}")
        lines.append("")
        
        for contrast_name, corrections in self.edge_tables.items():
            lines.append(f"Contrast: {contrast_name}")
            for correction, table in corrections.items():
                n_edges = len(table)
                lines.append(f"  {correction}: {n_edges} significant edges")
            lines.append("")
        
        return "\n".join(lines)
