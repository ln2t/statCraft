"""  
Data loader module for neuroimaging data.

This module handles:
- Data discovery using glob patterns
- Participant filtering
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
    Data loader for second-level neuroimaging analysis.
    
    Handles data discovery using flexible pattern matching, participant filtering,
    validation, and organization for group-level analyses.
    
    Parameters
    ----------
    bids_dir : str or Path
        Path to the dataset root directory. Only needs to contain participants.tsv
        for GLM and two-sample analyses.
    derivatives : list of str or Path
        List of paths to derivative folders containing images to analyze.
    output_dir : str or Path
        Path to output directory for StatCraft results.
    participants_file : str or Path, optional
        Path to a custom participants.tsv file. If not provided, will look for
        participants.tsv in bids_dir. Useful when bids_dir is directly a
        derivatives folder without participants.tsv. Only required for GLM and
        two-sample analyses.
    analysis_type : str, optional
        Type of analysis: 'one-sample', 'two-sample', 'paired', or 'glm'.
        Default is 'glm'. Only 'glm' and 'two-sample' require participants.tsv.
    
    Attributes
    ----------
    bids_dir : Path
        Path to dataset root directory.
    derivatives : list of Path
        Paths to derivative folders.
    output_dir : Path
        Output directory for results.
    participants_file : Path or None
        Path to custom participants file if provided.
    analysis_type : str
        Type of analysis being performed.
    layout : BIDSLayout
        PyBIDS layout object for data discovery.
    participants : pd.DataFrame or None
        Participants metadata from participants.tsv, or None if not required/available.
    """
    
    def __init__(
        self,
        bids_dir: Union[str, Path],
        derivatives: List[Union[str, Path]],
        output_dir: Union[str, Path],
        participants_file: Optional[Union[str, Path]] = None,
        analysis_type: str = "glm",
    ):
        self.bids_dir = Path(bids_dir)
        self.derivatives = [Path(d) for d in derivatives]
        self.output_dir = Path(output_dir)
        self.participants_file = Path(participants_file) if participants_file else None
        self.analysis_type = analysis_type
        
        # Validate paths
        self._validate_paths()
        
        # Initialize BIDS layout
        self.layout = self._init_bids_layout()
        
        # Load participants metadata (only if required by analysis type)
        self.participants = self._load_participants()
        
        # Cache for loaded images (NIfTI) and matrices (numpy)
        self._image_cache: Dict[str, Union[nib.Nifti1Image, np.ndarray]] = {}
        
        logger.info(f"DataLoader initialized with BIDS dir: {self.bids_dir}")
        if self.participants is not None:
            logger.info(f"Found {len(self.participants)} participants")
        else:
            logger.info("No participants file loaded (not required for this analysis type)")
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.bids_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.bids_dir}")
        
        # Check if participants.tsv is required for this analysis type
        requires_participants = self.analysis_type in ["glm", "two-sample", "two_sample"]
        
        if requires_participants:
            # Check for participants.tsv in bids_dir or use provided participants_file
            if self.participants_file:
                if not self.participants_file.exists():
                    raise FileNotFoundError(
                        f"Participants file not found: {self.participants_file}"
                    )
            else:
                participants_file = self.bids_dir / "participants.tsv"
                if not participants_file.exists():
                    raise FileNotFoundError(
                        f"participants.tsv not found in directory: {participants_file}. "
                        f"Use --participants-file to provide an alternative location."
                    )
        else:
            # For one-sample and paired analyses, participants.tsv is optional
            participants_file = self.bids_dir / "participants.tsv"
            if not self.participants_file and participants_file.exists():
                logger.debug("Found participants.tsv in BIDS dir (optional for this analysis type)")
        
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
    
    def _load_participants(self) -> Optional[pd.DataFrame]:
        """Load and validate participants.tsv if required."""
        # Check if participants.tsv is required for this analysis type
        requires_participants = self.analysis_type in ["glm", "two-sample", "two_sample"]
        
        # Try to find participants file
        if self.participants_file:
            participants_file = self.participants_file
        else:
            participants_file = self.bids_dir / "participants.tsv"
        
        # If file doesn't exist and it's not required, return None
        if not participants_file.exists():
            if not requires_participants:
                logger.debug(f"No participants.tsv found for {self.analysis_type} analysis (not required)")
                return None
            # If required and doesn't exist, error is raised in _validate_paths
            return None
        
        df = pd.read_csv(participants_file, sep="\t")
        
        # Ensure participant_id column exists
        if "participant_id" not in df.columns:
            raise ValueError("participants.tsv must have 'participant_id' column")
        
        # Remove 'sub-' prefix if present for easier matching
        df["subject"] = df["participant_id"].str.replace("sub-", "", regex=False)
        
        logger.info(f"Loaded participants.tsv from: {participants_file}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    
    def get_images(
        self,
        participant_label: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        extension: str = ".nii.gz",
    ) -> List[Dict]:
        """
        Get image files using pattern matching and optional participant filtering.

        Parameters
        ----------
        participant_label : list of str, optional
            List of participant labels to include (without 'sub-' prefix).
            If None, includes all participants.
        pattern : str, optional
            Glob pattern for finding files in derivatives.
            E.g., "**/*task-rest*space-MNI152*stat-effect*.nii.gz"
        exclude_pattern : str, optional
            Glob pattern to exclude files from the match.
            E.g., "*label-bad*"
        extension : str
            File extension to search for.

        Returns
        -------
        list of dict
            List of dictionaries with image info:
            - 'path': Path to the image file
            - 'subject': Subject ID
            - 'entities': Parsed BIDS-like entities
        """
        images = []

        # If pattern is provided, use glob matching
        if pattern:
            logger.info(f"Using glob pattern matching: {pattern}")
            glob_images = self._get_images_glob(pattern, participant_label, extension)
            images.extend(glob_images)
        else:
            # No pattern specified, try PyBIDS first
            logger.info("Using PyBIDS for image discovery")
            try:
                bids_images = self._get_images_pybids(participant_label, extension)
                if bids_images:
                    images.extend(bids_images)
            except Exception as e:
                logger.warning(f"PyBIDS image discovery failed: {e}")

            # If PyBIDS failed, fall back to glob
            if not images:
                logger.info("Falling back to glob pattern matching")
                glob_images = self._get_images_glob(pattern, participant_label, extension)
                images.extend(glob_images)

        # Apply exclude pattern if provided
        if exclude_pattern and images:
            logger.info(f"Applying exclude pattern: {exclude_pattern}")
            images = self._apply_exclude_pattern(images, exclude_pattern)
        
        if not images:
            if pattern:
                logger.warning(f"No images found matching pattern '{pattern}' in the following directories:")
                logger.warning(f"  - Dataset directory: {self.bids_dir}")
                for deriv_path in self.derivatives:
                    logger.warning(f"  - Derivatives: {deriv_path}")
                logger.warning("Hints:")
                logger.warning("  - Make sure the pattern uses quotes: '*.nii.gz' not *.nii.gz")
                logger.warning("  - Use '**/' for recursive search: '**/*.nii.gz'")
                logger.warning("  - Include task, session, space filters in the pattern:")
                logger.warning("    Example: '*task-rest*ses-01*space-MNI152*.nii.gz'")
            else:
                logger.warning("No images found matching the specified criteria")
        else:
            logger.info(f"Found {len(images)} images matching criteria")
        
        return images
    
    def _get_images_pybids(
        self,
        participant_label: Optional[List[str]],
        extension: str,
    ) -> List[Dict]:
        """Get images using PyBIDS."""
        images = []
        
        # Build query
        query = {"extension": extension.lstrip(".")}
        
        # Add participant filter
        if participant_label:
            # Remove 'sub-' prefix if present
            query["subject"] = [s.replace("sub-", "") for s in participant_label]
        
        # Get files from layout
        try:
            files = self.layout.get(**query)
            for f in files:
                entities = f.get_entities()
                file_path = Path(f.path)
                images.append({
                    "path": file_path,
                    "subject": entities.get("subject"),
                    "entities": entities,
                    "data_type": self._determine_data_type(file_path),
                })
        except Exception as e:
            logger.debug(f"PyBIDS query failed: {e}")
        
        return images
    
    def _get_images_glob(
        self,
        pattern: Optional[str],
        participant_label: Optional[List[str]],
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

        # Build list of search directories: bids_dir + all derivatives
        search_dirs = [self.bids_dir] + self.derivatives
        
        for search_path in search_dirs:
            logger.debug(f"Searching in: {search_path}")
            matches = list(search_path.glob(pattern))
            logger.debug(f"Found {len(matches)} potential matches in {search_path}")

            for img_path in matches:
                if not img_path.is_file():
                    continue

                # Parse BIDS-like entities from filename
                entities = self._parse_bids_entities(img_path)

                # Apply participant filter
                if participant_label:
                    subject = entities.get("subject")
                    if not subject:
                        logger.debug(f"Filtered out {img_path} (no subject entity)")
                        continue
                    # Normalize participant labels
                    normalized_labels = [p.replace("sub-", "") for p in participant_label]
                    if subject not in normalized_labels:
                        logger.debug(f"Filtered out {img_path} (subject {subject} not in participant_label)")
                        continue

                images.append({
                    "path": img_path,
                    "subject": entities.get("subject"),
                    "entities": entities,
                    "data_type": self._determine_data_type(img_path),
                })
        
        return images
    
    def _parse_bids_entities(self, filepath: Path) -> Dict:
        """Parse BIDS-like entities from filename."""
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
                # Map common keys
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
            Entity to pair by (e.g., "session", "task"). Must be present in filenames.
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
