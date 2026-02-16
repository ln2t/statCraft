"""
Cluster annotation module for anatomical labeling.

This module handles:
- Atlas-based cluster annotation
- Anatomical region labeling
- Coordinate system handling (MNI)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import get_data, coord_transform, resample_to_img

logger = logging.getLogger(__name__)


class ClusterAnnotator:
    """
    Annotate clusters with anatomical labels using atlases.
    
    Parameters
    ----------
    atlas : str
        Atlas to use for annotation. Options:
        - "harvard_oxford" (default): Harvard-Oxford cortical/subcortical
        - "aal": Automated Anatomical Labeling
        - "destrieux": Destrieux cortical atlas
        - "schaefer": Schaefer parcellation
        - Custom path to atlas NIfTI + labels file
    resolution : int
        Atlas resolution in mm (for atlases with multiple resolutions).
    
    Attributes
    ----------
    atlas_img : nibabel.Nifti1Image
        Atlas image.
    atlas_labels : dict
        Mapping of atlas indices to region names.
    """
    
    # Supported atlases
    SUPPORTED_ATLASES = [
        "harvard_oxford",
        "harvard_oxford_cortical",
        "harvard_oxford_subcortical",
        "aal",
        "destrieux",
        "schaefer",
    ]
    
    def __init__(
        self,
        atlas: str = "harvard_oxford",
        resolution: int = 2,
    ):
        self.atlas_name = atlas
        self.resolution = resolution
        self.atlas_img: Optional[nib.Nifti1Image] = None
        self.atlas_labels: Dict[int, str] = {}
        
        self._load_atlas()
    
    def _load_atlas(self) -> None:
        """Load the specified atlas."""
        logger.info(f"Loading atlas: {self.atlas_name}")
        
        if self.atlas_name in ["harvard_oxford", "harvard_oxford_cortical"]:
            self._load_harvard_oxford_cortical()
        elif self.atlas_name == "harvard_oxford_subcortical":
            self._load_harvard_oxford_subcortical()
        elif self.atlas_name == "aal":
            self._load_aal()
        elif self.atlas_name == "destrieux":
            self._load_destrieux()
        elif self.atlas_name.startswith("schaefer"):
            self._load_schaefer()
        elif Path(self.atlas_name).exists():
            self._load_custom_atlas(self.atlas_name)
        else:
            raise ValueError(
                f"Unknown atlas: {self.atlas_name}. "
                f"Supported: {self.SUPPORTED_ATLASES}"
            )
        
        logger.info(f"Loaded atlas with {len(self.atlas_labels)} regions")
    
    def _load_harvard_oxford_cortical(self) -> None:
        """Load Harvard-Oxford cortical atlas."""
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

        # Handle both string path and already-loaded Nifti1Image
        if isinstance(atlas["maps"], str):
            self.atlas_img = nib.load(atlas["maps"])
        else:
            self.atlas_img = atlas["maps"]

        # Labels start from 0 (background), then regions
        labels = atlas["labels"]
        self.atlas_labels = {i: label for i, label in enumerate(labels)}
    
    def _load_harvard_oxford_subcortical(self) -> None:
        """Load Harvard-Oxford subcortical atlas."""
        atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

        # Handle both string path and already-loaded Nifti1Image
        if isinstance(atlas["maps"], str):
            self.atlas_img = nib.load(atlas["maps"])
        else:
            self.atlas_img = atlas["maps"]

        labels = atlas["labels"]
        self.atlas_labels = {i: label for i, label in enumerate(labels)}
    
    def _load_aal(self) -> None:
        """Load AAL atlas."""
        atlas = datasets.fetch_atlas_aal()

        # Handle both string path and already-loaded Nifti1Image
        if isinstance(atlas["maps"], str):
            self.atlas_img = nib.load(atlas["maps"])
        else:
            self.atlas_img = atlas["maps"]

        # AAL has indices and labels
        indices = atlas["indices"]
        labels = atlas["labels"]
        self.atlas_labels = {int(idx): label for idx, label in zip(indices, labels)}
        self.atlas_labels[0] = "Background"
    
    def _load_destrieux(self) -> None:
        """Load Destrieux atlas."""
        atlas = datasets.fetch_atlas_destrieux_2009()

        # Handle both string path and already-loaded Nifti1Image
        if isinstance(atlas["maps"], str):
            self.atlas_img = nib.load(atlas["maps"])
        else:
            self.atlas_img = atlas["maps"]

        labels = atlas["labels"]
        # Destrieux labels are structured differently
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        self.atlas_labels = {i: str(label) for i, label in enumerate(labels)}
    
    def _load_schaefer(self) -> None:
        """Load Schaefer parcellation."""
        # Parse number of parcels from atlas name (e.g., "schaefer_400")
        n_parcels = 400  # default
        if "_" in self.atlas_name:
            try:
                n_parcels = int(self.atlas_name.split("_")[1])
            except (IndexError, ValueError):
                pass

        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_parcels,
            resolution_mm=self.resolution,
        )

        # Handle both string path and already-loaded Nifti1Image
        if isinstance(atlas["maps"], str):
            self.atlas_img = nib.load(atlas["maps"])
        else:
            self.atlas_img = atlas["maps"]

        labels = atlas["labels"]
        self.atlas_labels = {i + 1: label.decode() if isinstance(label, bytes) else str(label)
                           for i, label in enumerate(labels)}
        self.atlas_labels[0] = "Background"
    
    def _load_custom_atlas(self, atlas_path: str) -> None:
        """Load a custom atlas from file."""
        atlas_path = Path(atlas_path)
        
        if not atlas_path.exists():
            raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
        
        self.atlas_img = nib.load(atlas_path)
        
        # Look for labels file
        labels_path = atlas_path.with_suffix(".tsv")
        if not labels_path.exists():
            labels_path = atlas_path.with_suffix(".txt")
        
        if labels_path.exists():
            # Try to load labels
            try:
                labels_df = pd.read_csv(labels_path, sep="\t", header=None)
                if labels_df.shape[1] >= 2:
                    self.atlas_labels = dict(zip(labels_df[0], labels_df[1]))
                else:
                    self.atlas_labels = {i: label for i, label in enumerate(labels_df[0])}
            except Exception as e:
                logger.warning(f"Could not load labels file: {e}")
                # Generate numeric labels
                unique_vals = np.unique(get_data(self.atlas_img))
                self.atlas_labels = {int(v): f"Region_{int(v)}" for v in unique_vals}
        else:
            # Generate numeric labels
            unique_vals = np.unique(get_data(self.atlas_img))
            self.atlas_labels = {int(v): f"Region_{int(v)}" for v in unique_vals}
    
    def get_label_at_coord(
        self,
        coord: Tuple[float, float, float],
        coord_system: str = "mni",
    ) -> Tuple[int, str]:
        """
        Get the anatomical label at a specific coordinate.
        
        Parameters
        ----------
        coord : tuple of float
            (x, y, z) coordinate.
        coord_system : str
            Coordinate system: "mni" or "voxel".
        
        Returns
        -------
        tuple of (int, str)
            Atlas index and region label.
        """
        if self.atlas_img is None:
            raise ValueError("Atlas not loaded")
        
        x, y, z = coord
        
        # Convert MNI to voxel coordinates
        if coord_system == "mni":
            # Get inverse affine
            inv_affine = np.linalg.inv(self.atlas_img.affine)
            voxel_coord = coord_transform(x, y, z, inv_affine)
            i, j, k = [int(round(c)) for c in voxel_coord]
        else:
            i, j, k = int(x), int(y), int(z)
        
        # Get atlas data
        atlas_data = get_data(self.atlas_img)
        
        # Check bounds
        if not (0 <= i < atlas_data.shape[0] and 
                0 <= j < atlas_data.shape[1] and 
                0 <= k < atlas_data.shape[2]):
            return 0, "Outside atlas bounds"
        
        # Get atlas value
        atlas_value = int(atlas_data[i, j, k])
        label = self.atlas_labels.get(atlas_value, f"Unknown region ({atlas_value})")
        
        return atlas_value, label
    
    def annotate_cluster_table(
        self,
        cluster_table: pd.DataFrame,
        coord_columns: Tuple[str, str, str] = ("X", "Y", "Z"),
    ) -> pd.DataFrame:
        """
        Add anatomical labels to a cluster table.
        
        Parameters
        ----------
        cluster_table : pd.DataFrame
            Cluster table with peak coordinates.
        coord_columns : tuple of str
            Column names for X, Y, Z coordinates.
        
        Returns
        -------
        pd.DataFrame
            Cluster table with added "Region" column.
        """
        if len(cluster_table) == 0:
            cluster_table["Region"] = []
            return cluster_table
        
        # Get coordinate column names
        x_col, y_col, z_col = coord_columns
        
        # Check if columns exist (nilearn uses different names)
        if x_col not in cluster_table.columns:
            # Try alternative names
            possible_names = [
                ("X", "Y", "Z"),
                ("x", "y", "z"),
                ("Peak X", "Peak Y", "Peak Z"),
            ]
            for names in possible_names:
                if names[0] in cluster_table.columns:
                    x_col, y_col, z_col = names
                    break
            else:
                logger.warning("Could not find coordinate columns in cluster table")
                cluster_table["Region"] = ["Unknown"] * len(cluster_table)
                return cluster_table
        
        # Annotate each peak
        regions = []
        for _, row in cluster_table.iterrows():
            try:
                coord = (row[x_col], row[y_col], row[z_col])
                _, label = self.get_label_at_coord(coord)
                regions.append(label)
            except Exception as e:
                logger.warning(f"Could not annotate coordinate: {e}")
                regions.append("Unknown")
        
        # Add column
        annotated = cluster_table.copy()
        annotated["Region"] = regions
        
        return annotated
    
    def get_regions_in_mask(
        self,
        mask: Union[str, Path, nib.Nifti1Image],
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Get anatomical regions overlapping with a mask.
        
        Parameters
        ----------
        mask : str, Path, or nibabel.Nifti1Image
            Binary mask image.
        threshold : float
            Minimum proportion of region that must overlap.
        
        Returns
        -------
        dict
            Mapping of region names to overlap proportions.
        """
        if isinstance(mask, (str, Path)):
            mask = nib.load(mask)
        
        # Resample atlas to mask space if needed
        atlas_resampled = resample_to_img(
            self.atlas_img, mask, interpolation="nearest"
        )
        
        mask_data = get_data(mask) > 0
        atlas_data = get_data(atlas_resampled)
        
        # Count voxels per region in mask
        regions = {}
        for idx, label in self.atlas_labels.items():
            if idx == 0:  # Skip background
                continue
            
            region_mask = atlas_data == idx
            overlap = np.sum(mask_data & region_mask)
            region_size = np.sum(region_mask)
            
            if region_size > 0:
                proportion = overlap / region_size
                if proportion >= threshold:
                    regions[label] = proportion
        
        # Sort by overlap
        regions = dict(sorted(regions.items(), key=lambda x: x[1], reverse=True))
        
        return regions
    
    def create_region_summary(
        self,
        cluster_table: pd.DataFrame,
        stat_column: str = "Peak Stat",
    ) -> pd.DataFrame:
        """
        Create a summary of activations by anatomical region.
        
        Parameters
        ----------
        cluster_table : pd.DataFrame
            Annotated cluster table.
        stat_column : str
            Column name for statistical values.
        
        Returns
        -------
        pd.DataFrame
            Summary table grouped by region.
        """
        if "Region" not in cluster_table.columns:
            cluster_table = self.annotate_cluster_table(cluster_table)
        
        if len(cluster_table) == 0:
            return pd.DataFrame(columns=["Region", "N_Clusters", "Max_Stat", "Total_Voxels"])
        
        # Find stat column
        if stat_column not in cluster_table.columns:
            for col in cluster_table.columns:
                if "stat" in col.lower() or "peak" in col.lower():
                    stat_column = col
                    break
        
        # Find cluster size column
        size_column = None
        for col in cluster_table.columns:
            if "size" in col.lower() or "voxel" in col.lower() or "cluster" in col.lower():
                size_column = col
                break
        
        # Group by region
        summary = []
        for region, group in cluster_table.groupby("Region"):
            row = {
                "Region": region,
                "N_Clusters": len(group),
            }
            
            if stat_column in group.columns:
                row["Max_Stat"] = group[stat_column].max()
            
            if size_column and size_column in group.columns:
                row["Total_Voxels"] = group[size_column].sum()
            
            summary.append(row)
        
        summary_df = pd.DataFrame(summary)
        
        # Sort by number of clusters
        if len(summary_df) > 0:
            summary_df = summary_df.sort_values("N_Clusters", ascending=False)
        
        return summary_df.reset_index(drop=True)


def annotate_clusters(
    cluster_table: pd.DataFrame,
    atlas: str = "harvard_oxford",
) -> pd.DataFrame:
    """
    Convenience function to annotate a cluster table.
    
    Parameters
    ----------
    cluster_table : pd.DataFrame
        Cluster table with peak coordinates.
    atlas : str
        Atlas to use.
    
    Returns
    -------
    pd.DataFrame
        Annotated cluster table.
    """
    annotator = ClusterAnnotator(atlas=atlas)
    return annotator.annotate_cluster_table(cluster_table)
