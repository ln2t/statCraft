"""Tests for the design matrix module."""

import numpy as np
import pandas as pd
import pytest

from statcraft.core.design_matrix import DesignMatrixBuilder, create_contrast_from_string


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


class TestDesignMatrixBuilder:
    """Tests for DesignMatrixBuilder class."""
    
    def test_init(self, sample_participants):
        """Test initialization."""
        builder = DesignMatrixBuilder(sample_participants)
        assert builder.design_matrix is None
        assert len(builder.contrasts) == 0
    
    def test_one_sample_design_matrix(self, sample_participants):
        """Test one-sample design matrix creation."""
        builder = DesignMatrixBuilder(sample_participants)
        dm = builder.build_one_sample_design_matrix(n_subjects=6)
        
        assert dm.shape == (6, 1)
        assert "intercept" in dm.columns
        assert all(dm["intercept"] == 1.0)
    
    def test_two_sample_design_matrix(self, sample_participants):
        """Test two-sample design matrix creation."""
        builder = DesignMatrixBuilder(sample_participants)
        group_labels = ["A", "A", "A", "B", "B", "B"]
        
        dm = builder.build_two_sample_design_matrix(
            group_labels=group_labels,
            group_names=("groupA", "groupB"),
        )
        
        assert dm.shape == (6, 2)
        assert "groupA" in dm.columns
        assert "groupB" in dm.columns
        assert dm["groupA"].sum() == 3
        assert dm["groupB"].sum() == 3
    
    def test_paired_design_matrix(self, sample_participants):
        """Test paired design matrix creation."""
        builder = DesignMatrixBuilder(sample_participants)
        dm = builder.build_paired_design_matrix(n_pairs=10)
        
        assert dm.shape == (10, 1)
        assert "intercept" in dm.columns
    
    def test_group_design_matrix(self, sample_participants):
        """Test group design matrix creation."""
        builder = DesignMatrixBuilder(sample_participants)
        dm = builder.build_group_design_matrix("group")
        
        assert "patient" in dm.columns
        assert "control" in dm.columns
        assert dm["patient"].sum() == 3
        assert dm["control"].sum() == 3
    
    def test_design_matrix_with_columns(self, sample_participants):
        """Test design matrix with specified columns."""
        builder = DesignMatrixBuilder(sample_participants)
        dm = builder.build_design_matrix(
            columns=["age"],
            add_intercept=True,
            standardize_continuous=True,
        )
        
        assert "intercept" in dm.columns
        assert "age" in dm.columns
        # Age should be standardized (mean ~0, std ~1)
        assert abs(dm["age"].mean()) < 0.01
    
    def test_add_contrast_simple(self, sample_participants):
        """Test adding a simple contrast."""
        builder = DesignMatrixBuilder(sample_participants)
        builder.build_design_matrix(columns=["age"], add_intercept=True)
        
        name, vector = builder.add_contrast("age")
        
        assert "age" in name.lower() or name == "effectOfAge"
        assert len(vector) == 2
    
    def test_add_contrast_subtraction(self, sample_participants):
        """Test adding a subtraction contrast."""
        builder = DesignMatrixBuilder(sample_participants)
        builder.build_group_design_matrix("group")
        
        name, vector = builder.add_contrast("patient - control")
        
        assert len(vector) == 2
        # One should be positive, one negative
        assert 1.0 in vector or -1.0 in vector
    
    def test_add_contrasts_from_config(self, sample_participants):
        """Test adding multiple contrasts from config."""
        builder = DesignMatrixBuilder(sample_participants)
        builder.build_group_design_matrix("group")
        
        contrasts = builder.add_contrasts_from_config([
            "patient",
            "control",
            {"expression": "patient - control", "name": "patients_vs_controls"},
        ])
        
        assert len(contrasts) == 3
        assert "patients_vs_controls" in contrasts
    
    def test_generate_contrast_name(self, sample_participants):
        """Test automatic contrast name generation."""
        builder = DesignMatrixBuilder(sample_participants)
        
        # Simple variable
        name = builder._generate_contrast_name("age")
        assert "age" in name.lower()
        
        # Subtraction
        name = builder._generate_contrast_name("groupA - groupB")
        assert "versus" in name.lower() or "vs" in name.lower()
    
    def test_summary(self, sample_participants):
        """Test summary generation."""
        builder = DesignMatrixBuilder(sample_participants)
        builder.build_one_sample_design_matrix(n_subjects=6)
        
        summary = builder.summary()
        
        assert "Design Matrix" in summary
        assert "intercept" in summary


class TestContrastUtility:
    """Tests for create_contrast_from_string utility."""
    
    def test_create_contrast(self, sample_participants):
        """Test contrast creation utility function."""
        builder = DesignMatrixBuilder(sample_participants)
        dm = builder.build_group_design_matrix("group")
        
        name, vector = create_contrast_from_string("patient - control", dm)
        
        assert len(vector) == 2
