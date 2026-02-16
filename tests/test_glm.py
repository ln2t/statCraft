"""Tests for the GLM module."""

import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import pytest


class TestSecondLevelGLM:
    """Tests for SecondLevelGLM class."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample NIfTI images for testing."""
        images = []
        shape = (20, 20, 20)
        affine = np.eye(4)
        affine[0, 0] = 2
        affine[1, 1] = 2
        affine[2, 2] = 2
        
        for i in range(10):
            # Create random data with some structure
            data = np.random.normal(100, 10, shape)
            # Add a "signal" that varies by subject
            data[5:15, 5:15, 5:15] += i * 2
            
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            images.append(img)
        
        return images
    
    @pytest.fixture
    def sample_design_matrix(self):
        """Create sample design matrix."""
        return pd.DataFrame({
            "intercept": np.ones(10),
            "age": np.random.normal(30, 5, 10),
        })
    
    def test_glm_init(self):
        """Test GLM initialization."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        
        assert glm.model is None
        assert len(glm.results) == 0
    
    def test_glm_fit(self, sample_images, sample_design_matrix):
        """Test GLM fitting."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        glm.fit(sample_images, sample_design_matrix)
        
        assert glm.model is not None
    
    def test_glm_compute_contrast(self, sample_images, sample_design_matrix):
        """Test computing contrasts."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        glm.fit(sample_images, sample_design_matrix)
        
        results = glm.compute_contrast([1, 0], contrast_name="intercept_test")
        
        assert "stat" in results
        assert isinstance(results["stat"], nib.Nifti1Image)
    
    def test_one_sample_ttest(self, sample_images):
        """Test one-sample t-test."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        results = glm.one_sample_ttest(sample_images, contrast_name="mean_effect")
        
        assert "stat" in results
        assert "one_sample" in glm.results or "mean_effect" in glm.results
    
    def test_two_sample_ttest(self, sample_images):
        """Test two-sample t-test."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        
        # Split images into two groups
        group1 = sample_images[:5]
        group2 = sample_images[5:]
        
        results = glm.two_sample_ttest(
            group1, group2,
            group1_name="groupA",
            group2_name="groupB",
        )
        
        assert "stat" in results
    
    def test_paired_ttest(self, sample_images):
        """Test paired t-test."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        
        # Split images into pairs
        condition1 = sample_images[:5]
        condition2 = sample_images[5:]
        
        results = glm.paired_ttest(
            condition1, condition2,
            condition1_name="pre",
            condition2_name="post",
        )
        
        assert "stat" in results
    
    def test_save_results(self, sample_images, sample_design_matrix):
        """Test saving results."""
        from statcraft.core.glm import SecondLevelGLM
        
        glm = SecondLevelGLM()
        glm.fit(sample_images, sample_design_matrix)
        glm.compute_contrast([1, 0], contrast_name="test_contrast")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = glm.save_results(tmpdir)
            
            assert len(saved) > 0
            for path in saved.values():
                assert Path(path).exists()


class TestNormalizationMixin:
    """Tests for NormalizationMixin class."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for normalization."""
        images = []
        shape = (20, 20, 20)
        affine = np.eye(4)
        
        for i in range(5):
            # Create data with different means
            data = np.random.normal(100 + i * 10, 10, shape)
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            images.append(img)
        
        return images
    
    def test_mean_normalize(self, sample_images):
        """Test mean normalization."""
        from statcraft.core.glm import NormalizationMixin
        
        normalized = NormalizationMixin.mean_normalize(sample_images)
        
        assert len(normalized) == len(sample_images)
        
        # All normalized images should have similar mean
        means = [np.mean(img.get_fdata()[img.get_fdata() != 0]) for img in normalized]
        assert np.std(means) < 10  # Means should be similar (around 100)
    
    def test_compute_group_mask(self, sample_images):
        """Test group mask computation."""
        from statcraft.core.glm import NormalizationMixin
        
        mask = NormalizationMixin.compute_group_mask(sample_images, threshold=0.5)
        
        assert isinstance(mask, nib.Nifti1Image)
        assert mask.shape == sample_images[0].shape[:3]
