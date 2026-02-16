"""Tests for the inference module."""

import numpy as np
import nibabel as nib
import pandas as pd
import pytest


class TestStatisticalInference:
    """Tests for StatisticalInference class."""
    
    @pytest.fixture
    def sample_stat_map(self):
        """Create a sample statistical map."""
        # Create a simple 3D array with some "activation"
        data = np.zeros((20, 20, 20))
        
        # Add a cluster of activation
        data[5:10, 5:10, 5:10] = 4.0  # High t-values
        data[15:18, 15:18, 15:18] = 3.5  # Another cluster
        
        # Add some noise
        data += np.random.normal(0, 0.1, data.shape)
        
        # Create affine (simple MNI-like)
        affine = np.eye(4)
        affine[0, 0] = 2  # 2mm voxels
        affine[1, 1] = 2
        affine[2, 2] = 2
        affine[:3, 3] = [-20, -20, -20]  # Origin offset
        
        return nib.Nifti1Image(data, affine)
    
    def test_inference_initialization(self):
        """Test StatisticalInference initialization."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(
            alpha=0.05,
            height_threshold=0.001,
            cluster_threshold=10,
        )
        
        assert inference.alpha == 0.05
        assert inference.height_threshold == 0.001
        assert inference.cluster_threshold == 10
        assert inference.two_sided == True
    
    def test_uncorrected_threshold(self, sample_stat_map):
        """Test uncorrected thresholding."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(
            height_threshold=0.01,  # Less strict for testing
            cluster_threshold=5,
        )
        
        thresholded, table = inference.threshold_uncorrected(
            sample_stat_map,
            contrast_name="test",
        )
        
        assert thresholded is not None
        assert isinstance(table, pd.DataFrame)
    
    def test_fdr_threshold(self, sample_stat_map):
        """Test FDR thresholding."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(
            alpha=0.05,
            cluster_threshold=5,
        )
        
        thresholded, table = inference.threshold_fdr(
            sample_stat_map,
            contrast_name="test",
        )
        
        assert thresholded is not None
        assert isinstance(table, pd.DataFrame)
    
    def test_bonferroni_threshold(self, sample_stat_map):
        """Test Bonferroni thresholding."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(
            alpha=0.05,
            cluster_threshold=5,
        )
        
        thresholded, table = inference.threshold_fwer_bonferroni(
            sample_stat_map,
            contrast_name="test",
        )
        
        assert thresholded is not None
        assert isinstance(table, pd.DataFrame)
    
    def test_run_all_corrections(self, sample_stat_map):
        """Test running all corrections."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(
            alpha=0.05,
            height_threshold=0.01,
            cluster_threshold=5,
        )
        
        results = inference.run_all_corrections(
            sample_stat_map,
            contrast_name="test",
            include_permutation=False,
        )
        
        assert "uncorrected" in results
        assert "fdr" in results
        assert "bonferroni" in results
    
    def test_get_cluster_table(self, sample_stat_map):
        """Test retrieving cluster tables."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(cluster_threshold=5)
        inference.threshold_uncorrected(sample_stat_map, contrast_name="test")
        
        table = inference.get_cluster_table("test", "uncorrected")
        
        assert isinstance(table, pd.DataFrame)
    
    def test_get_thresholded_map(self, sample_stat_map):
        """Test retrieving thresholded maps."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(cluster_threshold=5)
        inference.threshold_uncorrected(sample_stat_map, contrast_name="test")
        
        thresholded = inference.get_thresholded_map("test", "uncorrected")
        
        assert isinstance(thresholded, nib.Nifti1Image)
    
    def test_summary(self, sample_stat_map):
        """Test summary generation."""
        from statcraft.core.inference import StatisticalInference
        
        inference = StatisticalInference(cluster_threshold=5)
        inference.threshold_uncorrected(sample_stat_map, contrast_name="test")
        
        summary = inference.summary()
        
        assert "Statistical Inference" in summary
        assert "test" in summary
