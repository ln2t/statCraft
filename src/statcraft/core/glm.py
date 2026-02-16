"""
Second-level GLM module for neuroimaging analysis.

This module handles:
- GLM fitting with custom design matrices
- One-sample, two-sample, and paired t-tests
- Contrast computation
- Result extraction
- Connectivity matrix analysis (edge-wise GLM)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import mean_img, math_img, concat_imgs, new_img_like
from nilearn.masking import compute_brain_mask, apply_mask, unmask, intersect_masks
from scipy import stats

logger = logging.getLogger(__name__)


class SecondLevelGLM:
    """
    Second-level GLM for group-level neuroimaging analysis.
    
    Supports GLM analysis with custom design matrices, as well as
    one-sample, two-sample, and paired t-tests.
    
    Parameters
    ----------
    mask : str, Path, or nibabel image, optional
        Brain mask for analysis. If None, computed from data.
    smoothing_fwhm : float, optional
        Smoothing kernel FWHM in mm. If None, no smoothing.
    
    Attributes
    ----------
    model : SecondLevelModel
        Fitted nilearn SecondLevelModel.
    results : dict
        Dictionary of results for each contrast.
    """
    
    def __init__(
        self,
        mask: Optional[Union[str, Path, nib.Nifti1Image]] = None,
        smoothing_fwhm: Optional[float] = None,
    ):
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.model: Optional[SecondLevelModel] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self._images: Optional[List] = None
        self._design_matrix: Optional[pd.DataFrame] = None
    
    def fit(
        self,
        images: List[Union[str, Path, nib.Nifti1Image]],
        design_matrix: pd.DataFrame,
    ) -> "SecondLevelGLM":
        """
        Fit the second-level GLM.
        
        Parameters
        ----------
        images : list
            List of image paths or nibabel images.
        design_matrix : pd.DataFrame
            Design matrix with rows = images, columns = regressors.
        
        Returns
        -------
        SecondLevelGLM
            Self, for method chaining.
        """
        # Convert paths to strings
        images_list = []
        for img in images:
            if isinstance(img, (str, Path)):
                images_list.append(str(img))
            else:
                images_list.append(img)
        
        # Validate dimensions
        if len(images_list) != len(design_matrix):
            raise ValueError(
                f"Number of images ({len(images_list)}) does not match "
                f"design matrix rows ({len(design_matrix)})"
            )
        
        logger.info(f"Fitting GLM with {len(images_list)} images")
        logger.info(f"Design matrix columns: {list(design_matrix.columns)}")
        
        # Create and fit model
        self.model = SecondLevelModel(
            mask_img=self.mask,
            smoothing_fwhm=self.smoothing_fwhm,
        )
        
        self.model.fit(images_list, design_matrix=design_matrix)
        
        self._images = images_list
        self._design_matrix = design_matrix
        
        logger.info("GLM fitted successfully")
        return self
    
    def compute_contrast(
        self,
        contrast: Union[str, np.ndarray, List[float]],
        contrast_name: Optional[str] = None,
        stat_type: str = "t",
        output_type: str = "all",
    ) -> Dict[str, nib.Nifti1Image]:
        """
        Compute a contrast.

        Parameters
        ----------
        contrast : str, array, or list
            Contrast specification. Can be:
            - Column name from design matrix
            - Contrast vector (array or list)
            - Contrast expression (e.g., "group1 - group2")
        contrast_name : str, optional
            Name for the contrast. Auto-generated if None.
        stat_type : str
            Type of statistic: "t" or "F".
        output_type : str
            What to compute: "stat", "p_value", "effect_size", "effect_variance", or "all".

        Returns
        -------
        dict
            Dictionary with requested statistical maps.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate contrast name if needed
        if contrast_name is None:
            if isinstance(contrast, str):
                contrast_name = contrast.replace(" ", "").replace("-", "Vs")
            else:
                contrast_name = f"contrast_{len(self.results)}"

        logger.info(f"Computing contrast: {contrast_name}")

        results = {}

        if output_type == "all":
            output_types = ["z_score", "stat", "p_value", "effect_size", "effect_variance"]
        else:
            output_types = [output_type]

        for otype in output_types:
            try:
                stat_map = self.model.compute_contrast(
                    second_level_contrast=contrast,
                    second_level_stat_type=stat_type,
                    output_type=otype,
                )
                results[otype] = stat_map
            except Exception as e:
                logger.warning(f"Could not compute {otype}: {e}")

        # Ensure we have at least the stat map
        if "stat" not in results:
            # If no specific output_type worked, try without it (default behavior)
            try:
                logger.info("Trying compute_contrast with default parameters")
                stat_map = self.model.compute_contrast(
                    second_level_contrast=contrast,
                    second_level_stat_type=stat_type,
                )
                results["stat"] = stat_map
            except Exception as e:
                raise ValueError(f"Failed to compute contrast: {e}")

        # Store results
        self.results[contrast_name] = {
            "contrast_def": contrast,
            "stat_type": stat_type,
            "maps": results,
        }

        return results
    
    def one_sample_ttest(
        self,
        images: List[Union[str, Path, nib.Nifti1Image]],
        contrast_name: str = "one_sample",
    ) -> Dict[str, nib.Nifti1Image]:
        """
        Perform a one-sample t-test.
        
        Tests whether the mean activation is significantly different from zero.
        
        Parameters
        ----------
        images : list
            List of image paths or nibabel images.
        contrast_name : str
            Name for the contrast.
        
        Returns
        -------
        dict
            Dictionary with statistical maps.
        """
        # Create simple design matrix with intercept
        design_matrix = pd.DataFrame({"intercept": np.ones(len(images))})
        
        # Fit model
        self.fit(images, design_matrix)
        
        # Compute contrast
        return self.compute_contrast(
            [1.0],
            contrast_name=contrast_name,
        )
    
    def two_sample_ttest(
        self,
        images_group1: List[Union[str, Path, nib.Nifti1Image]],
        images_group2: List[Union[str, Path, nib.Nifti1Image]],
        group1_name: str = "group1",
        group2_name: str = "group2",
        contrast_name: Optional[str] = None,
    ) -> Dict[str, nib.Nifti1Image]:
        """
        Perform a two-sample t-test.
        
        Tests whether the mean activation differs between two groups.
        
        Parameters
        ----------
        images_group1 : list
            Images from group 1.
        images_group2 : list
            Images from group 2.
        group1_name : str
            Name for group 1.
        group2_name : str
            Name for group 2.
        contrast_name : str, optional
            Name for the contrast. Default: "{group1_name}_vs_{group2_name}".
        
        Returns
        -------
        dict
            Dictionary with statistical maps.
        """
        # Combine images
        all_images = list(images_group1) + list(images_group2)
        
        # Create design matrix
        n1 = len(images_group1)
        n2 = len(images_group2)
        
        design_matrix = pd.DataFrame({
            group1_name: [1.0] * n1 + [0.0] * n2,
            group2_name: [0.0] * n1 + [1.0] * n2,
        })
        
        # Fit model
        self.fit(all_images, design_matrix)
        
        # Compute contrast (group1 - group2)
        if contrast_name is None:
            contrast_name = f"{group1_name}_vs_{group2_name}"
        
        return self.compute_contrast(
            [1.0, -1.0],
            contrast_name=contrast_name,
        )
    
    def paired_ttest(
        self,
        images_condition1: List[Union[str, Path, nib.Nifti1Image]],
        images_condition2: List[Union[str, Path, nib.Nifti1Image]],
        condition1_name: str = "condition1",
        condition2_name: str = "condition2",
        contrast_name: Optional[str] = None,
    ) -> Dict[str, nib.Nifti1Image]:
        """
        Perform a paired t-test.

        Tests whether the mean difference between conditions is different from zero.

        Parameters
        ----------
        images_condition1 : list
            Images from condition 1 (one per subject).
        images_condition2 : list
            Images from condition 2 (one per subject, same order).
        condition1_name : str
            Name for condition 1.
        condition2_name : str
            Name for condition 2.
        contrast_name : str, optional
            Name for the contrast.

        Returns
        -------
        dict
            Dictionary with statistical maps.
        """
        if len(images_condition1) != len(images_condition2):
            raise ValueError(
                f"Number of images in condition 1 ({len(images_condition1)}) "
                f"does not match condition 2 ({len(images_condition2)})"
            )

        n_pairs = len(images_condition1)
        logger.info(f"Performing paired t-test with {n_pairs} pairs")

        # Compute difference images using memory-mapped loading
        diff_images = []
        for img1, img2 in zip(images_condition1, images_condition2):
            # Load images with memory mapping
            if isinstance(img1, (str, Path)):
                img1 = nib.load(img1, mmap=True)
            if isinstance(img2, (str, Path)):
                img2 = nib.load(img2, mmap=True)

            # Compute difference using dataobj (memory-mapped) and float32 to save memory
            diff_data = np.asarray(img1.dataobj, dtype=np.float32) - np.asarray(img2.dataobj, dtype=np.float32)
            diff_img = nib.Nifti1Image(diff_data, img1.affine, img1.header)
            diff_images.append(diff_img)

        # One-sample t-test on differences
        if contrast_name is None:
            contrast_name = f"{condition1_name}_vs_{condition2_name}"

        return self.one_sample_ttest(diff_images, contrast_name=contrast_name)
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        Save all results to disk.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory.
        prefix : str
            Prefix for output filenames.
        
        Returns
        -------
        dict
            Dictionary mapping result names to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for contrast_name, result in self.results.items():
            for map_type, stat_map in result["maps"].items():
                # Create filename
                if prefix:
                    filename = f"{prefix}_{contrast_name}_{map_type}.nii.gz"
                else:
                    filename = f"{contrast_name}_{map_type}.nii.gz"
                
                filepath = output_dir / filename
                nib.save(stat_map, filepath)
                saved_files[f"{contrast_name}_{map_type}"] = filepath
                logger.debug(f"Saved: {filepath}")
        
        logger.info(f"Saved {len(saved_files)} result files to {output_dir}")
        return saved_files
    
    def get_stat_map(self, contrast_name: str) -> nib.Nifti1Image:
        """Get the t-statistic map for a contrast."""
        if contrast_name not in self.results:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        return self.results[contrast_name]["maps"]["stat"]
    
    def get_p_map(self, contrast_name: str) -> nib.Nifti1Image:
        """Get the p-value map for a contrast."""
        if contrast_name not in self.results:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        return self.results[contrast_name]["maps"]["p_value"]
    
    def get_effect_size_map(self, contrast_name: str) -> nib.Nifti1Image:
        """Get the effect size map for a contrast."""
        if contrast_name not in self.results:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        return self.results[contrast_name]["maps"]["effect_size"]


class NormalizationMixin:
    """
    Mixin class providing image normalization methods.
    """
    
    @staticmethod
    def mean_normalize(
        images: List[Union[str, Path, nib.Nifti1Image]],
        mask: Optional[Union[str, Path, nib.Nifti1Image]] = None,
    ) -> List[nib.Nifti1Image]:
        """
        Normalize images by their mean signal.

        Parameters
        ----------
        images : list
            List of images to normalize.
        mask : str, Path, or nibabel image, optional
            Mask defining the region for mean computation.

        Returns
        -------
        list
            List of normalized images.
        """
        normalized = []

        # Load mask if provided (with memory mapping)
        if mask is not None:
            if isinstance(mask, (str, Path)):
                mask = nib.load(mask, mmap=True)
            mask_data = np.asarray(mask.dataobj, dtype=bool) > 0
        else:
            mask_data = None

        for img in images:
            if isinstance(img, (str, Path)):
                img = nib.load(img, mmap=True)

            # Use memory-mapped data access with float32 to save memory
            data = np.asarray(img.dataobj, dtype=np.float32)

            # Compute mean
            if mask_data is not None:
                mean_val = np.mean(data[mask_data])
            else:
                mean_val = np.mean(data[data != 0])

            # Normalize
            if mean_val != 0:
                norm_data = data / mean_val * 100  # Percent signal
            else:
                norm_data = data

            norm_img = nib.Nifti1Image(norm_data, img.affine, img.header)
            normalized.append(norm_img)

        logger.info(f"Mean-normalized {len(normalized)} images")
        return normalized
    
    @staticmethod
    def compute_group_mask(
        images: List[Union[str, Path, nib.Nifti1Image]],
        threshold: float = 0.5,
    ) -> nib.Nifti1Image:
        """
        Compute a group brain mask from multiple images.

        Parameters
        ----------
        images : list
            List of images.
        threshold : float
            Proportion of subjects that must have signal for a voxel to be included.

        Returns
        -------
        nibabel.Nifti1Image
            Group brain mask.
        """
        # Load first image for reference (with memory mapping)
        if isinstance(images[0], (str, Path)):
            ref_img = nib.load(images[0], mmap=True)
        else:
            ref_img = images[0]

        # Compute individual masks with memory-efficient loading
        mask_sum = np.zeros(ref_img.shape[:3], dtype=np.float32)

        for img in images:
            if isinstance(img, (str, Path)):
                img = nib.load(img, mmap=True)

            # Simple mask: non-zero voxels, using memory-mapped access
            data = np.asarray(img.dataobj, dtype=np.float32)
            mask_sum += (np.abs(data) > 0).astype(np.float32)

        # Threshold
        group_mask_data = (mask_sum / len(images)) >= threshold

        group_mask = nib.Nifti1Image(
            group_mask_data.astype(np.int8),
            ref_img.affine,
            ref_img.header,
        )

        logger.info(
            f"Computed group mask with {np.sum(group_mask_data)} voxels "
            f"({threshold * 100:.0f}% threshold)"
        )

        return group_mask


class ConnectivityGLM:
    """
    Edge-wise GLM for connectivity matrix analysis.
    
    Performs second-level analysis on connectivity matrices by running
    a separate GLM for each edge (upper triangle of the matrix).
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    results : dict
        Dictionary of results for each contrast.
    n_rois : int
        Number of ROIs in the connectivity matrices.
    n_edges : int
        Number of edges (upper triangle elements).
    """
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.n_rois: Optional[int] = None
        self.n_edges: Optional[int] = None
        self._design_matrix: Optional[pd.DataFrame] = None
        self._matrices: Optional[np.ndarray] = None
        self._triu_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def _vectorize_upper_triangle(
        self,
        matrices: List[np.ndarray],
    ) -> np.ndarray:
        """
        Vectorize the upper triangle of connectivity matrices.
        
        Parameters
        ----------
        matrices : list of np.ndarray
            List of 2D connectivity matrices (n_rois x n_rois).
        
        Returns
        -------
        np.ndarray
            2D array of shape (n_subjects, n_edges).
        """
        n_subjects = len(matrices)
        n_rois = matrices[0].shape[0]
        
        # Get upper triangle indices (excluding diagonal)
        self._triu_indices = np.triu_indices(n_rois, k=1)
        n_edges = len(self._triu_indices[0])
        
        # Extract upper triangles
        edges = np.zeros((n_subjects, n_edges), dtype=np.float64)
        for i, matrix in enumerate(matrices):
            edges[i, :] = matrix[self._triu_indices]
        
        return edges
    
    def _reconstruct_matrix(
        self,
        edge_values: np.ndarray,
        fill_diagonal: float = 0.0,
    ) -> np.ndarray:
        """
        Reconstruct a symmetric matrix from upper triangle values.
        
        Parameters
        ----------
        edge_values : np.ndarray
            1D array of edge values.
        fill_diagonal : float
            Value to fill on the diagonal.
        
        Returns
        -------
        np.ndarray
            Symmetric 2D matrix (n_rois x n_rois).
        """
        matrix = np.zeros((self.n_rois, self.n_rois), dtype=np.float64)
        matrix[self._triu_indices] = edge_values
        matrix = matrix + matrix.T  # Make symmetric
        np.fill_diagonal(matrix, fill_diagonal)
        return matrix
    
    def fit(
        self,
        matrices: List[np.ndarray],
        design_matrix: pd.DataFrame,
    ) -> "ConnectivityGLM":
        """
        Fit the edge-wise GLM.
        
        Parameters
        ----------
        matrices : list of np.ndarray
            List of 2D connectivity matrices.
        design_matrix : pd.DataFrame
            Design matrix with rows = subjects, columns = regressors.
        
        Returns
        -------
        ConnectivityGLM
            Self, for method chaining.
        """
        # Validate dimensions
        if len(matrices) != len(design_matrix):
            raise ValueError(
                f"Number of matrices ({len(matrices)}) does not match "
                f"design matrix rows ({len(design_matrix)})"
            )
        
        # Store dimensions
        self.n_rois = matrices[0].shape[0]
        
        # Vectorize matrices
        edges = self._vectorize_upper_triangle(matrices)
        self.n_edges = edges.shape[1]
        
        self._matrices = edges
        self._design_matrix = design_matrix
        
        logger.info(f"Fitting edge-wise GLM with {len(matrices)} subjects")
        logger.info(f"Matrix size: {self.n_rois}x{self.n_rois}, {self.n_edges} edges")
        logger.info(f"Design matrix columns: {list(design_matrix.columns)}")
        
        return self
    
    def compute_contrast(
        self,
        contrast: Union[np.ndarray, List[float]],
        contrast_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute a contrast using edge-wise t-tests.
        
        Parameters
        ----------
        contrast : array or list
            Contrast vector.
        contrast_name : str, optional
            Name for the contrast.
        
        Returns
        -------
        dict
            Dictionary with:
            - 't_matrix': t-statistic matrix (n_rois x n_rois)
            - 'p_matrix': p-value matrix (n_rois x n_rois)
            - 't_edges': t-statistics for each edge (1D)
            - 'p_edges': p-values for each edge (1D)
        """
        if self._matrices is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if contrast_name is None:
            contrast_name = f"contrast_{len(self.results)}"
        
        contrast = np.array(contrast, dtype=np.float64)
        X = self._design_matrix.values.astype(np.float64)
        Y = self._matrices  # (n_subjects, n_edges)
        
        n_subjects = X.shape[0]
        n_regressors = X.shape[1]
        df = n_subjects - n_regressors
        
        logger.info(f"Computing contrast: {contrast_name}")
        logger.info(f"Degrees of freedom: {df}")
        
        # Compute GLM: beta = (X'X)^-1 X' Y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            logger.warning("Design matrix is singular, using pseudo-inverse")
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        beta = XtX_inv @ X.T @ Y  # (n_regressors, n_edges)
        
        # Compute residuals and MSE
        Y_pred = X @ beta
        residuals = Y - Y_pred
        mse = np.sum(residuals ** 2, axis=0) / df  # (n_edges,)
        
        # Compute contrast estimates
        c_beta = contrast @ beta  # (n_edges,)
        
        # Compute standard error of contrast
        c_var = contrast @ XtX_inv @ contrast  # scalar
        se = np.sqrt(c_var * mse)  # (n_edges,)
        
        # Compute t-statistics
        t_edges = np.divide(c_beta, se, out=np.zeros_like(c_beta), where=se != 0)
        
        # Compute p-values (two-tailed)
        p_edges = 2 * stats.t.sf(np.abs(t_edges), df)
        
        # Reconstruct matrices
        t_matrix = self._reconstruct_matrix(t_edges, fill_diagonal=0.0)
        p_matrix = self._reconstruct_matrix(p_edges, fill_diagonal=1.0)
        
        results = {
            't_matrix': t_matrix,
            'p_matrix': p_matrix,
            't_edges': t_edges,
            'p_edges': p_edges,
        }
        
        # Store results
        self.results[contrast_name] = {
            'contrast_def': contrast.tolist(),
            'maps': results,
            'n_edges': self.n_edges,
            'n_rois': self.n_rois,
            'df': df,
        }
        
        logger.info(f"Computed t-statistics for {self.n_edges} edges")
        
        return results
    
    def one_sample_ttest(
        self,
        matrices: List[np.ndarray],
        contrast_name: str = "one_sample",
    ) -> Dict[str, np.ndarray]:
        """
        Perform a one-sample t-test on connectivity matrices.
        
        Tests whether the mean connectivity is significantly different from zero.
        
        Parameters
        ----------
        matrices : list of np.ndarray
            List of connectivity matrices.
        contrast_name : str
            Name for the contrast.
        
        Returns
        -------
        dict
            Dictionary with statistical matrices.
        """
        design_matrix = pd.DataFrame({"intercept": np.ones(len(matrices))})
        self.fit(matrices, design_matrix)
        return self.compute_contrast([1.0], contrast_name=contrast_name)
    
    def two_sample_ttest(
        self,
        matrices_group1: List[np.ndarray],
        matrices_group2: List[np.ndarray],
        group1_name: str = "group1",
        group2_name: str = "group2",
        contrast_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform a two-sample t-test on connectivity matrices.
        
        Parameters
        ----------
        matrices_group1 : list of np.ndarray
            Connectivity matrices from group 1.
        matrices_group2 : list of np.ndarray
            Connectivity matrices from group 2.
        group1_name : str
            Name for group 1.
        group2_name : str
            Name for group 2.
        contrast_name : str, optional
            Name for the contrast.
        
        Returns
        -------
        dict
            Dictionary with statistical matrices.
        """
        all_matrices = list(matrices_group1) + list(matrices_group2)
        n1 = len(matrices_group1)
        n2 = len(matrices_group2)
        
        design_matrix = pd.DataFrame({
            group1_name: [1.0] * n1 + [0.0] * n2,
            group2_name: [0.0] * n1 + [1.0] * n2,
        })
        
        self.fit(all_matrices, design_matrix)
        
        if contrast_name is None:
            contrast_name = f"{group1_name}_vs_{group2_name}"
        
        return self.compute_contrast([1.0, -1.0], contrast_name=contrast_name)
    
    def paired_ttest(
        self,
        matrices_condition1: List[np.ndarray],
        matrices_condition2: List[np.ndarray],
        condition1_name: str = "condition1",
        condition2_name: str = "condition2",
        contrast_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform a paired t-test on connectivity matrices.
        
        Parameters
        ----------
        matrices_condition1 : list of np.ndarray
            Connectivity matrices from condition 1.
        matrices_condition2 : list of np.ndarray
            Connectivity matrices from condition 2 (same order as condition 1).
        condition1_name : str
            Name for condition 1.
        condition2_name : str
            Name for condition 2.
        contrast_name : str, optional
            Name for the contrast.
        
        Returns
        -------
        dict
            Dictionary with statistical matrices.
        """
        if len(matrices_condition1) != len(matrices_condition2):
            raise ValueError(
                f"Number of matrices in condition 1 ({len(matrices_condition1)}) "
                f"does not match condition 2 ({len(matrices_condition2)})"
            )
        
        n_pairs = len(matrices_condition1)
        logger.info(f"Performing paired t-test with {n_pairs} pairs")
        
        # Compute difference matrices
        diff_matrices = [
            m1 - m2 for m1, m2 in zip(matrices_condition1, matrices_condition2)
        ]
        
        if contrast_name is None:
            contrast_name = f"{condition1_name}_vs_{condition2_name}"
        
        return self.one_sample_ttest(diff_matrices, contrast_name=contrast_name)
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        bids_prefix: str = "",
        roi_names: Optional[List[str]] = None,
        roi_coordinates: Optional[np.ndarray] = None,
        atlas_name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Save all results to disk with BIDS-compatible naming.
        
        Output structure:
        - output_dir/
            - data/           # Statistical maps (.npy) and JSON sidecars
            - (rest of output structure handled by caller)
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory (should be root; data/ will be created inside).
        bids_prefix : str
            BIDS prefix for output filenames (e.g., 'sub-001_task-rest').
        roi_names : list of str, optional
            ROI names for the JSON sidecar.
        roi_coordinates : np.ndarray, optional
            ROI coordinates for the JSON sidecar.
        atlas_name : str, optional
            Atlas name for the JSON sidecar.
        
        Returns
        -------
        dict
            Dictionary mapping result names to file paths.
        """
        import json
        
        output_dir = Path(output_dir)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for contrast_name, result in self.results.items():
            # Create BIDS-compatible filenames
            if bids_prefix:
                t_filename = f"{bids_prefix}_contrast-{contrast_name}_stat-tstat.npy"
                p_filename = f"{bids_prefix}_contrast-{contrast_name}_stat-pval.npy"
                t_json_filename = f"{bids_prefix}_contrast-{contrast_name}_stat-tstat.json"
            else:
                t_filename = f"contrast-{contrast_name}_stat-tstat.npy"
                p_filename = f"contrast-{contrast_name}_stat-pval.npy"
                t_json_filename = f"contrast-{contrast_name}_stat-tstat.json"
            
            t_filepath = data_dir / t_filename
            p_filepath = data_dir / p_filename
            t_json_filepath = data_dir / t_json_filename
            
            # Save t-statistic matrix
            np.save(t_filepath, result['maps']['t_matrix'])
            
            # Save p-value matrix
            np.save(p_filepath, result['maps']['p_matrix'])
            
            # Create JSON sidecar for t-stat map
            sidecar = {
                'contrast_name': contrast_name,
                'n_rois': result['n_rois'],
                'n_edges': result['n_edges'],
                'df': result['df'],
                'type': 't-statistic',
                'description': 'Edge-wise t-statistics from GLM analysis',
            }
            if roi_names is not None:
                sidecar['ROINames'] = roi_names
            if roi_coordinates is not None:
                sidecar['ROICoordinates'] = roi_coordinates.tolist()
            if atlas_name is not None:
                sidecar['AtlasName'] = atlas_name
            
            with open(t_json_filepath, 'w') as f:
                json.dump(sidecar, f, indent=2)
            
            saved_files[f"{contrast_name}_tstat"] = t_filepath
            saved_files[f"{contrast_name}_pval"] = p_filepath
            saved_files[f"{contrast_name}_tstat_json"] = t_json_filepath
            
            logger.debug(f"Saved: {t_filepath}")
            logger.debug(f"Saved: {p_filepath}")
            logger.debug(f"Saved: {t_json_filepath}")
        
        logger.info(f"Saved {len(saved_files)} connectivity GLM result files to {data_dir}")
        return saved_files
    
    def get_t_matrix(self, contrast_name: str) -> np.ndarray:
        """Get the t-statistic matrix for a contrast."""
        if contrast_name not in self.results:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        return self.results[contrast_name]['maps']['t_matrix']
    
    def get_p_matrix(self, contrast_name: str) -> np.ndarray:
        """Get the p-value matrix for a contrast."""
        if contrast_name not in self.results:
            raise KeyError(f"Contrast '{contrast_name}' not found")
        return self.results[contrast_name]['maps']['p_matrix']
