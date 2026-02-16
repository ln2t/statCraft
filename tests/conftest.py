"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_bids_dir(temp_dir):
    """Create a minimal BIDS-like directory structure."""
    bids_dir = temp_dir / "bids"
    bids_dir.mkdir()
    
    # Create participants.tsv
    participants = pd.DataFrame({
        "participant_id": ["sub-01", "sub-02", "sub-03", "sub-04"],
        "age": [25, 30, 28, 35],
        "group": ["patient", "control", "patient", "control"],
        "sex": ["M", "F", "M", "F"],
    })
    participants.to_csv(bids_dir / "participants.tsv", sep="\t", index=False)
    
    # Create minimal subject directories
    for sub in ["sub-01", "sub-02", "sub-03", "sub-04"]:
        (bids_dir / sub).mkdir()
    
    return bids_dir


@pytest.fixture
def sample_derivatives_dir(temp_dir):
    """Create a minimal derivatives directory with NIfTI files."""
    deriv_dir = temp_dir / "derivatives"
    deriv_dir.mkdir()
    
    try:
        import nibabel as nib
        
        # Create sample NIfTI files for each subject
        shape = (20, 20, 20)
        affine = np.eye(4)
        affine[0, 0] = 2  # 2mm voxels
        affine[1, 1] = 2
        affine[2, 2] = 2
        
        for i, sub in enumerate(["sub-01", "sub-02", "sub-03", "sub-04"]):
            sub_dir = deriv_dir / sub / "func"
            sub_dir.mkdir(parents=True)
            
            # Create random data with some structure
            data = np.random.normal(100, 10, shape).astype(np.float32)
            data[5:15, 5:15, 5:15] += 50 + i * 10  # Add "activation"
            
            img = nib.Nifti1Image(data, affine)
            nib.save(img, sub_dir / f"{sub}_task-nback_space-MNI_stat-effect_statmap.nii.gz")
    
    except ImportError:
        # If nibabel not available, just create empty files
        for sub in ["sub-01", "sub-02", "sub-03", "sub-04"]:
            sub_dir = deriv_dir / sub / "func"
            sub_dir.mkdir(parents=True)
            (sub_dir / f"{sub}_task-nback_space-MNI_stat-effect_statmap.nii.gz").touch()
    
    return deriv_dir


@pytest.fixture
def sample_nifti_image():
    """Create a sample NIfTI image."""
    try:
        import nibabel as nib
        
        shape = (20, 20, 20)
        data = np.random.normal(100, 10, shape).astype(np.float32)
        
        affine = np.eye(4)
        affine[0, 0] = 2
        affine[1, 1] = 2
        affine[2, 2] = 2
        
        return nib.Nifti1Image(data, affine)
    except ImportError:
        pytest.skip("nibabel not installed")


@pytest.fixture
def sample_stat_map():
    """Create a sample statistical map with activations."""
    try:
        import nibabel as nib
        
        shape = (20, 20, 20)
        data = np.zeros(shape, dtype=np.float32)
        
        # Add activation clusters
        data[5:10, 5:10, 5:10] = 4.0  # High t-values
        data[15:18, 15:18, 15:18] = 3.5  # Another cluster
        
        # Add some noise
        data += np.random.normal(0, 0.1, shape).astype(np.float32)
        
        affine = np.eye(4)
        affine[0, 0] = 2
        affine[1, 1] = 2
        affine[2, 2] = 2
        affine[:3, 3] = [-20, -20, -20]  # Origin offset
        
        return nib.Nifti1Image(data, affine)
    except ImportError:
        pytest.skip("nibabel not installed")


@pytest.fixture
def sample_participants():
    """Create sample participant data."""
    return pd.DataFrame({
        "participant_id": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"],
        "subject": ["01", "02", "03", "04", "05", "06"],
        "age": [25, 30, 28, 35, 22, 40],
        "group": ["patient", "control", "patient", "control", "patient", "control"],
        "sex": ["M", "F", "M", "F", "F", "M"],
    })
