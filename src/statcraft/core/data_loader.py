"""
Data loader module for BIDS-compliant neuroimaging data.

This module handles:
- BIDS data discovery and filtering
- MNI space validation
- Data consistency checks
- Paired test data organization
- Connectivity matrix (.npy) loading
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from bids import BIDSLayout

logger = logging.getLogger(__name__)


class DataLoader:
    """
    BIDS-compliant data loader for second-level neuroimaging analysis.
    
    Handles data discovery, filtering, validation, and organization
    for group-level analyses.
    
    Parameters
    ----------
    bids_dir : str or Path
        Path to the BIDS rawdata directory (must contain participants.tsv).
    derivatives : list of str or Path
        List of paths to derivative folders containing images to analyze.
    output_dir : str or Path
        Path to output directory for StatCraft results.
    
    Attributes
    ----------
    bids_dir : Path
        Path to BIDS rawdata directory.
    derivatives : list of Path
        Paths to derivative folders.
    output_dir : Path
        Output directory for results.
    layout : BIDSLayout
        PyBIDS layout object for data discovery.
    participants : pd.DataFrame
        Participants metadata from participants.tsv.
    """
    
    def __init__(
        self,
        bids_dir: Union[str, Path],
        derivatives: List[Union[str, Path]],
        output_dir: Union[str, Path],
    ):
        self.bids_dir = Path(bids_dir)
        self.derivatives = [Path(d) for d in derivatives]
        self.output_dir = Path(output_dir)
        
        # Validate paths
        self._validate_paths()
        
        # Initialize BIDS layout
        self.layout = self._init_bids_layout()
        
        # Load participants metadata
        self.participants = self._load_participants()
        
        # Cache for loaded images (NIfTI) and matrices (numpy)
        self._image_cache: Dict[str, Union[nib.Nifti1Image, np.ndarray]] = {}
        
        logger.info(f"DataLoader initialized with BIDS dir: {self.bids_dir}")
        logger.info(f"Found {len(self.participants)} participants")
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.bids_dir.exists():
            raise FileNotFoundError(f"BIDS directory not found: {self.bids_dir}")
        
        participants_file = self.bids_dir / "participants.tsv"
        if not participants_file.exists():
            raise FileNotFoundError(
                f"participants.tsv not found in BIDS directory: {participants_file}"
            )
        
        for deriv_path in self.derivatives:
            if not deriv_path.exists():
                raise FileNotFoundError(f"Derivative directory not found: {deriv_path}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _determine_data_type(filepath: Path) -> str:
        """
        Determine the data type based on file extension.

        Parameters
        ----------
        filepath : Path
            Path to the data file.

        Returns
        -------
        str
            Data type: 'nifti' for .nii/.nii.gz files, 'connectivity' for .npy files.
        """
        suffix = filepath.suffix.lower()
        name = filepath.name.lower()
        
        if suffix == '.npy':
            return 'connectivity'
        elif suffix == '.gz' and name.endswith('.nii.gz'):
            return 'nifti'
        elif suffix == '.nii':
            return 'nifti'
        else:
            # Default to nifti for backwards compatibility
            return 'nifti'
    
    def _init_bids_layout(self) -> BIDSLayout:
        """Initialize PyBIDS layout with derivatives."""
        try:
            layout = BIDSLayout(
                self.bids_dir,
                derivatives=self.derivatives,
                validate=False,  # Allow non-standard derivatives
            )
            return layout
        except Exception as e:
            logger.warning(f"Could not initialize full BIDS layout: {e}")
            # Fallback: initialize without derivatives validation
            layout = BIDSLayout(self.bids_dir, validate=False)
            return layout
    
    def _load_participants(self) -> pd.DataFrame:
        """Load and validate participants.tsv."""
        participants_file = self.bids_dir / "participants.tsv"
        df = pd.read_csv(participants_file, sep="\t")
        
        # Ensure participant_id column exists
        if "participant_id" not in df.columns:
            raise ValueError("participants.tsv must have 'participant_id' column")
        
        # Remove 'sub-' prefix if present for easier matching
        df["subject"] = df["participant_id"].str.replace("sub-", "", regex=False)
        
        logger.info(f"Loaded participants.tsv with columns: {list(df.columns)}")
        return df
    
    def get_images(
        self,
        bids_filters: Optional[Dict] = None,
        pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        extension: str = ".nii.gz",
    ) -> List[Dict]:
        """
        Get image files matching BIDS filters.

        Parameters
        ----------
        bids_filters : dict, optional
            Dictionary of BIDS entities to filter by.
            E.g., {"task": "nback", "session": "pre", "subject": ["01", "02"]}
        pattern : str, optional
            Glob pattern for finding files in derivatives.
            E.g., "**/sub-*_task-*_stat-effect_statmap.nii.gz"
        exclude_pattern : str, optional
            Glob pattern to exclude files from the match.
            E.g., "*label-GS*"
        extension : str
            File extension to search for.

        Returns
        -------
        list of dict
            List of dictionaries with image info:
            - 'path': Path to the image file
            - 'subject': Subject ID
            - 'session': Session ID (if applicable)
            - 'task': Task name (if applicable)
            - 'entities': All BIDS entities
        """
        images = []
        bids_filters = bids_filters or {}

        # If pattern is provided, ONLY use glob matching (more specific)
        if pattern:
            logger.info(f"Using glob pattern matching: {pattern}")
            glob_images = self._get_images_glob(pattern, bids_filters, extension)
            images.extend(glob_images)
        else:
            # No pattern specified, try PyBIDS first
            logger.info("Using PyBIDS for image discovery")
            try:
                bids_images = self._get_images_pybids(bids_filters, extension)
                if bids_images:
                    images.extend(bids_images)
            except Exception as e:
                logger.warning(f"PyBIDS image discovery failed: {e}")

            # If PyBIDS failed, fall back to glob
            if not images:
                logger.info("Falling back to glob pattern matching")
                glob_images = self._get_images_glob(pattern, bids_filters, extension)
                images.extend(glob_images)

        # Apply exclude pattern if provided
        if exclude_pattern and images:
            logger.info(f"Applying exclude pattern: {exclude_pattern}")
            images = self._apply_exclude_pattern(images, exclude_pattern)
        
        if not images:
            if pattern:
                logger.warning(f"No images found matching pattern '{pattern}' in derivatives directories:")
                for deriv_path in self.derivatives:
                    logger.warning(f"  - {deriv_path}")
                logger.warning("Hints:")
                logger.warning("  - Make sure the pattern uses quotes: '*.nii.gz' not *.nii.gz")
                logger.warning("  - Use '**/' for recursive search: '**/*.nii.gz'")

                # Suggest including BIDS filters in pattern if filters were provided
                if bids_filters:
                    logger.warning("  - Include BIDS filters in the pattern instead of using separate flags:")
                    if "task" in bids_filters:
                        task_val = bids_filters["task"]
                        logger.warning(f"    Example: '*task-{task_val}*.nii.gz' instead of --task {task_val}")
                    if "session" in bids_filters:
                        session_val = bids_filters["session"]
                        logger.warning(f"    Example: '*ses-{session_val}*.nii.gz'")
            else:
                logger.warning("No images found matching the specified criteria")
        else:
            logger.info(f"Found {len(images)} images matching criteria")
        
        return images
    
    def _get_images_pybids(
        self,
        bids_filters: Dict,
        extension: str,
    ) -> List[Dict]:
        """Get images using PyBIDS."""
        images = []
        
        # Build query
        query = {"extension": extension.lstrip(".")}
        
        # Add filters
        if "subject" in bids_filters:
            subjects = bids_filters["subject"]
            if isinstance(subjects, str):
                subjects = [subjects]
            # Remove 'sub-' prefix if present
            query["subject"] = [s.replace("sub-", "") for s in subjects]
        
        if "session" in bids_filters:
            sessions = bids_filters["session"]
            if isinstance(sessions, str):
                sessions = [sessions]
            query["session"] = [s.replace("ses-", "") for s in sessions]
        
        if "task" in bids_filters:
            tasks = bids_filters["task"]
            if isinstance(tasks, str):
                tasks = [tasks]
            query["task"] = tasks
        
        # Get files from layout
        try:
            files = self.layout.get(**query)
            for f in files:
                entities = f.get_entities()
                file_path = Path(f.path)
                images.append({
                    "path": file_path,
                    "subject": entities.get("subject"),
                    "session": entities.get("session"),
                    "task": entities.get("task"),
                    "entities": entities,
                    "data_type": self._determine_data_type(file_path),
                })
        except Exception as e:
            logger.debug(f"PyBIDS query failed: {e}")
        
        return images
    
    def _get_images_glob(
        self,
        pattern: Optional[str],
        bids_filters: Dict,
        extension: str,
    ) -> List[Dict]:
        """Get images using glob pattern matching."""
        images = []

        # Default pattern
        if pattern is None:
            pattern = f"**/*{extension}"

        # If pattern doesn't start with ** and doesn't contain /, prepend ** for recursive search
        if not pattern.startswith("**/") and "/" not in pattern:
            logger.info(f"Pattern '{pattern}' doesn't specify subdirectories, adding '**/' for recursive search")
            pattern = f"**/{pattern}"

        logger.debug(f"Searching with glob pattern: {pattern}")

        for deriv_path in self.derivatives:
            logger.debug(f"Searching in: {deriv_path}")
            matches = list(deriv_path.glob(pattern))
            logger.debug(f"Found {len(matches)} potential matches in {deriv_path}")

            for img_path in matches:
                if not img_path.is_file():
                    continue

                # Parse BIDS entities from filename
                entities = self._parse_bids_entities(img_path)

                # Apply filters
                if not self._matches_filters(entities, bids_filters):
                    logger.debug(f"Filtered out {img_path} (doesn't match filters)")
                    continue

                images.append({
                    "path": img_path,
                    "subject": entities.get("subject"),
                    "session": entities.get("session"),
                    "task": entities.get("task"),
                    "entities": entities,
                    "data_type": self._determine_data_type(img_path),
                })
        
        return images
    
    def _parse_bids_entities(self, filepath: Path) -> Dict:
        """Parse BIDS entities from filename."""
        entities = {}
        name = filepath.stem
        
        # Remove extension(s)
        while "." in name:
            name = name.rsplit(".", 1)[0]
        
        # Parse key-value pairs
        parts = name.split("_")
        for part in parts:
            if "-" in part:
                key, value = part.split("-", 1)
                # Map common BIDS keys
                key_map = {
                    "sub": "subject",
                    "ses": "session",
                    "task": "task",
                    "run": "run",
                    "space": "space",
                    "desc": "description",
                    "stat": "stat",
                }
                key = key_map.get(key, key)
                entities[key] = value
        
        return entities
    
    def _matches_filters(self, entities: Dict, filters: Dict) -> bool:
        """Check if entities match the specified filters."""
        for key, values in filters.items():
            if key not in entities:
                continue

            if isinstance(values, str):
                values = [values]

            # Normalize values (remove prefixes)
            normalized_values = []
            for v in values:
                if key == "subject":
                    v = v.replace("sub-", "")
                elif key == "session":
                    v = v.replace("ses-", "")
                normalized_values.append(v)

            if entities[key] not in normalized_values:
                return False

        return True

    def _apply_exclude_pattern(self, images: List[Dict], exclude_pattern: str) -> List[Dict]:
        """
        Apply exclude pattern to filter out unwanted images.

        Parameters
        ----------
        images : list of dict
            List of image info dictionaries.
        exclude_pattern : str
            Glob pattern to exclude files.

        Returns
        -------
        list of dict
            Filtered list of images.
        """
        from fnmatch import fnmatch

        filtered_images = []
        excluded_count = 0

        for img in images:
            # Get the filename to match against the pattern
            filename = img["path"].name

            # Check if the filename matches the exclude pattern
            if fnmatch(filename, exclude_pattern):
                logger.debug(f"Excluding {img['path']} (matches exclude pattern)")
                excluded_count += 1
            else:
                filtered_images.append(img)

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} image(s) matching pattern '{exclude_pattern}'")

        return filtered_images
    
    def load_image(self, path: Union[str, Path], use_cache: bool = True) -> nib.Nifti1Image:
        """
        Load a NIfTI image from path with optional memory mapping.

        Parameters
        ----------
        path : str or Path
            Path to the NIfTI image.
        use_cache : bool
            Whether to use the image cache. Default True.

        Returns
        -------
        nibabel.Nifti1Image
            Loaded image.
        """
        path = Path(path)
        path_str = str(path)

        if use_cache and path_str in self._image_cache:
            return self._image_cache[path_str]

        # Load with memory mapping to reduce memory usage
        img = nib.load(path, mmap=True)

        if use_cache:
            self._image_cache[path_str] = img

        return img

    def clear_cache(self) -> None:
        """
        Clear the image cache to free memory.

        This should be called after validation or when images are no longer needed.
        """
        n_cached = len(self._image_cache)
        self._image_cache.clear()
        logger.info(f"Cleared image cache ({n_cached} images)")

    def load_connectivity_matrix(
        self,
        path: Union[str, Path],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Load a connectivity matrix from a .npy file.

        Parameters
        ----------
        path : str or Path
            Path to the .npy file containing the connectivity matrix.
        use_cache : bool
            Whether to use the cache. Default True.

        Returns
        -------
        np.ndarray
            2D connectivity matrix (n_rois x n_rois).
        """
        path = Path(path)
        path_str = str(path)

        if use_cache and path_str in self._image_cache:
            return self._image_cache[path_str]

        matrix = np.load(path)

        if use_cache:
            self._image_cache[path_str] = matrix

        return matrix

    def load_connectivity_sidecar(
        self,
        matrix_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Load the JSON sidecar for a connectivity matrix.

        The sidecar file should have the same name as the .npy file
        but with .json extension. It contains metadata including:
        - ROICoordinates: list of [x, y, z] MNI coordinates
        - ROINames: list of region names
        - AtlasName: name of the atlas used

        Parameters
        ----------
        matrix_path : str or Path
            Path to the .npy connectivity matrix file.

        Returns
        -------
        dict
            JSON sidecar contents with connectivity metadata.
        """
        matrix_path = Path(matrix_path)
        json_path = matrix_path.with_suffix('.json')

        if not json_path.exists():
            logger.warning(f"No JSON sidecar found for {matrix_path}")
            return {}

        with open(json_path, 'r') as f:
            sidecar = json.load(f)

        return sidecar

    def get_connectivity_metadata(
        self,
        image_info: Dict,
    ) -> Dict[str, Any]:
        """
        Extract connectivity-specific metadata from an image info dict.

        Loads the JSON sidecar and extracts ROI coordinates and names.

        Parameters
        ----------
        image_info : dict
            Image info dictionary with 'path' key.

        Returns
        -------
        dict
            Connectivity metadata including:
            - roi_coordinates: np.ndarray of shape (n_rois, 3)
            - roi_names: list of str
            - atlas_name: str
        """
        sidecar = self.load_connectivity_sidecar(image_info['path'])

        metadata = {
            'roi_coordinates': None,
            'roi_names': None,
            'atlas_name': None,
        }

        if 'ROICoordinates' in sidecar:
            coords = sidecar['ROICoordinates']
            metadata['roi_coordinates'] = np.array(coords)
            logger.debug(f"Loaded {len(coords)} ROI coordinates")

        if 'ROINames' in sidecar:
            metadata['roi_names'] = sidecar['ROINames']
            logger.debug(f"Loaded {len(sidecar['ROINames'])} ROI names")

        if 'AtlasName' in sidecar:
            metadata['atlas_name'] = sidecar['AtlasName']
            logger.debug(f"Atlas: {sidecar['AtlasName']}")

        return metadata

    def validate_connectivity_matrices(
        self,
        images: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate connectivity matrices for consistency.

        Checks:
        - Matrix is 2D and square
        - Matrix shape is consistent across all images
        - Matrix is symmetric (within tolerance)

        Parameters
        ----------
        images : list of dict
            List of image info dictionaries (must be connectivity type).

        Returns
        -------
        tuple of (list, list)
            Valid images and invalid images with reasons.
        """
        valid = []
        invalid = []
        reference_shape = None

        for img_info in images:
            try:
                matrix = self.load_connectivity_matrix(img_info['path'])

                # Check 2D
                if matrix.ndim != 2:
                    invalid.append({
                        **img_info,
                        'reason': f"Matrix is not 2D: shape {matrix.shape}",
                    })
                    continue

                # Check square
                if matrix.shape[0] != matrix.shape[1]:
                    invalid.append({
                        **img_info,
                        'reason': f"Matrix is not square: shape {matrix.shape}",
                    })
                    continue

                # Check symmetry (tolerance for numerical precision)
                if not np.allclose(matrix, matrix.T, atol=1e-6):
                    invalid.append({
                        **img_info,
                        'reason': "Matrix is not symmetric",
                    })
                    continue

                # Store first matrix shape as reference
                if reference_shape is None:
                    reference_shape = matrix.shape

                # Check shape consistency
                if matrix.shape != reference_shape:
                    invalid.append({
                        **img_info,
                        'reason': f"Shape mismatch: {matrix.shape} vs {reference_shape}",
                    })
                    continue

                valid.append(img_info)

            except Exception as e:
                invalid.append({
                    **img_info,
                    'reason': f"Failed to load: {str(e)}",
                })

        if invalid:
            logger.warning(f"{len(invalid)} connectivity matrices failed validation")
            for img in invalid:
                logger.warning(f"  {img['path']}: {img['reason']}")

        logger.info(f"{len(valid)} connectivity matrices passed validation")
        return valid, invalid
    
    def validate_mni_space(
        self,
        images: List[Dict],
        reference_shape: Tuple[int, ...] = (91, 109, 91),
        tolerance: float = 0.1,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate that images are in MNI space with consistent dimensions.
        
        Parameters
        ----------
        images : list of dict
            List of image info dictionaries.
        reference_shape : tuple
            Expected image shape for MNI space (default: 2mm resolution).
        tolerance : float
            Tolerance for affine comparison.
        
        Returns
        -------
        tuple of (list, list)
            Valid images and invalid images with reasons.
        """
        valid = []
        invalid = []
        reference_affine = None
        nan_inf_warning_shown = False

        for img_info in images:
            try:
                img = self.load_image(img_info["path"])

                # Check for NaN/Inf using memory-mapped access
                # Sample the data instead of loading it all for efficiency
                data_sample = np.asarray(img.dataobj[:10, :10, :10], dtype=np.float32)
                if np.any(np.isnan(data_sample)) or np.any(np.isinf(data_sample)):
                    if not nan_inf_warning_shown:
                        logger.warning(f"Image contains NaN/Inf: {img_info['path']}")
                        logger.warning("Further NaN/Inf warnings will be suppressed. See logs for details.")
                        nan_inf_warning_shown = True
                    else:
                        # Log to debug level for detailed tracking without cluttering output
                        logger.debug(f"Image contains NaN/Inf: {img_info['path']}")

                # Store first image as reference
                if reference_affine is None:
                    reference_affine = img.affine
                    reference_shape = img.shape[:3]

                # Check shape consistency
                if img.shape[:3] != reference_shape:
                    invalid.append({
                        **img_info,
                        "reason": f"Shape mismatch: {img.shape[:3]} vs {reference_shape}",
                    })
                    continue

                # Check affine consistency
                if not np.allclose(img.affine, reference_affine, atol=tolerance):
                    invalid.append({
                        **img_info,
                        "reason": "Affine mismatch with reference image",
                    })
                    continue

                valid.append(img_info)

            except Exception as e:
                invalid.append({
                    **img_info,
                    "reason": f"Failed to load: {str(e)}",
                })
        
        if invalid:
            logger.warning(f"{len(invalid)} images failed validation")
            for img in invalid:
                logger.warning(f"  {img['path']}: {img['reason']}")
        
        logger.info(f"{len(valid)} images passed validation")
        return valid, invalid
    
    def organize_paired_data(
        self,
        images: List[Dict],
        pair_by: str,
        condition1: str,
        condition2: str,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Organize images into pairs for paired t-tests.
        
        Parameters
        ----------
        images : list of dict
            List of image info dictionaries.
        pair_by : str
            BIDS entity to pair by (e.g., "session", "task").
        condition1 : str
            Value of pair_by for first condition.
        condition2 : str
            Value of pair_by for second condition.
        
        Returns
        -------
        list of tuple
            List of (condition1_image, condition2_image) tuples.
        """
        pairs = []
        
        # Group images by subject
        by_subject: Dict[str, Dict[str, Dict]] = {}
        
        for img in images:
            subject = img["subject"]
            if subject is None:
                logger.warning(f"Image has no subject: {img['path']}")
                continue
            
            condition = img["entities"].get(pair_by)
            if condition is None:
                logger.warning(f"Image has no {pair_by}: {img['path']}")
                continue
            
            if subject not in by_subject:
                by_subject[subject] = {}
            
            # Normalize condition values
            condition_normalized = condition.replace(f"{pair_by[:3]}-", "")
            by_subject[subject][condition_normalized] = img
        
        # Create pairs
        for subject, conditions in by_subject.items():
            cond1_norm = condition1.replace(f"{pair_by[:3]}-", "")
            cond2_norm = condition2.replace(f"{pair_by[:3]}-", "")
            
            if cond1_norm in conditions and cond2_norm in conditions:
                pairs.append((conditions[cond1_norm], conditions[cond2_norm]))
            else:
                missing = []
                if cond1_norm not in conditions:
                    missing.append(condition1)
                if cond2_norm not in conditions:
                    missing.append(condition2)
                logger.warning(
                    f"Subject {subject} missing conditions: {missing}"
                )
        
        logger.info(f"Created {len(pairs)} pairs for {pair_by}: {condition1} vs {condition2}")
        return pairs
    
    def get_participants_for_images(
        self,
        images: List[Dict],
    ) -> pd.DataFrame:
        """
        Get participant metadata for a list of images.
        
        Parameters
        ----------
        images : list of dict
            List of image info dictionaries.
        
        Returns
        -------
        pd.DataFrame
            Participant metadata for the specified images, in order.
            
        Raises
        ------
        ValueError
            If any subject in images is not found in participants.tsv.
        """
        subjects = [img["subject"] for img in images]
        
        # Check for missing subjects first
        missing_subjects = []
        for subject in set(subjects):
            mask = self.participants["subject"] == subject
            if not mask.any():
                missing_subjects.append(subject)
        
        if missing_subjects:
            missing_subjects.sort()
            error_msg = (
                f"Found {len(missing_subjects)} subject(s) in images that are missing from participants.tsv:\n"
                f"  {', '.join(missing_subjects)}\n\n"
                f"This indicates an incomplete dataset. Please ensure all subjects in your "
                f"derivative images are present in {self.bids_dir}/participants.tsv before proceeding.\n\n"
                f"You can either:\n"
                f"  1. Add the missing subjects to participants.tsv, or\n"
                f"  2. Remove the derivative files for these subjects"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create ordered DataFrame
        result = []
        for subject in subjects:
            mask = self.participants["subject"] == subject
            result.append(self.participants[mask].iloc[0])
        
        return pd.DataFrame(result).reset_index(drop=True)
