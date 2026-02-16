"""Integration tests for the full pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestPipelineIntegration:
    """Integration tests for StatCraftPipeline."""
    
    @pytest.fixture
    def setup_bids_data(self, temp_dir):
        """Set up BIDS-compliant test data."""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel not installed")
        
        bids_dir = temp_dir / "bids"
        bids_dir.mkdir()
        
        # Create participants.tsv
        participants = pd.DataFrame({
            "participant_id": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"],
            "age": [25, 30, 28, 35, 22, 40],
            "group": ["patient", "patient", "patient", "control", "control", "control"],
            "sex": ["M", "F", "M", "F", "F", "M"],
        })
        participants.to_csv(bids_dir / "participants.tsv", sep="\t", index=False)
        
        # Create derivatives with NIfTI files
        deriv_dir = temp_dir / "derivatives"
        
        shape = (20, 20, 20)
        affine = np.eye(4)
        affine[0, 0] = 2
        affine[1, 1] = 2
        affine[2, 2] = 2
        
        for i, sub in enumerate(["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]):
            sub_dir = deriv_dir / sub / "func"
            sub_dir.mkdir(parents=True)
            
            # Create data with group difference
            data = np.random.normal(100, 5, shape).astype(np.float32)
            
            # Add activation that differs by group
            if i < 3:  # Patients
                data[5:15, 5:15, 5:15] += 30
            else:  # Controls
                data[5:15, 5:15, 5:15] += 10
            
            img = nib.Nifti1Image(data, affine)
            nib.save(img, sub_dir / f"{sub}_task-nback_stat-effect_statmap.nii.gz")
        
        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        return {
            "bids_dir": bids_dir,
            "deriv_dir": deriv_dir,
            "output_dir": output_dir,
        }
    
    def test_one_sample_pipeline(self, setup_bids_data):
        """Test one-sample t-test pipeline."""
        from statcraft.pipeline import StatCraftPipeline
        
        data = setup_bids_data
        
        pipeline = StatCraftPipeline(
            bids_dir=data["bids_dir"],
            output_dir=data["output_dir"],
            derivatives=[data["deriv_dir"]],
            config={
                "analysis_type": "one-sample",
                "file_pattern": "**/*_task-nback_stat-effect_statmap.nii.gz",
            },
        )
        
        results = pipeline.run()
        
        assert "images" in results
        assert len(results["images"]) == 6
        assert "glm_results" in results
        assert "saved_files" in results
    
    def test_two_sample_pipeline(self, setup_bids_data):
        """Test two-sample t-test pipeline."""
        from statcraft.pipeline import StatCraftPipeline
        
        data = setup_bids_data
        
        pipeline = StatCraftPipeline(
            bids_dir=data["bids_dir"],
            output_dir=data["output_dir"],
            derivatives=[data["deriv_dir"]],
            config={
                "analysis_type": "two-sample",
                "group_comparison": {
                    "group_column": "group",
                    "group1": "patient",
                    "group2": "control",
                },
                "file_pattern": "**/*_task-nback_stat-effect_statmap.nii.gz",
            },
        )
        
        results = pipeline.run()
        
        assert "images" in results
        assert len(results["images"]) == 6
        assert "glm_results" in results
    
    def test_glm_pipeline_with_covariates(self, setup_bids_data):
        """Test GLM pipeline with covariates."""
        from statcraft.pipeline import StatCraftPipeline
        
        data = setup_bids_data
        
        pipeline = StatCraftPipeline(
            bids_dir=data["bids_dir"],
            output_dir=data["output_dir"],
            derivatives=[data["deriv_dir"]],
            config={
                "analysis_type": "glm",
                "design_matrix": {
                    "columns": ["age"],
                    "add_intercept": True,
                },
                "contrasts": ["age"],
                "file_pattern": "**/*_task-nback_stat-effect_statmap.nii.gz",
            },
        )
        
        results = pipeline.run()
        
        assert "images" in results
        assert "design_matrix" in results
        assert "contrasts" in results
        assert "effectOfAge" in results["contrasts"] or "age" in str(results["contrasts"])
    
    def test_report_generation(self, setup_bids_data):
        """Test HTML report generation."""
        from statcraft.pipeline import StatCraftPipeline
        
        data = setup_bids_data
        
        pipeline = StatCraftPipeline(
            bids_dir=data["bids_dir"],
            output_dir=data["output_dir"],
            derivatives=[data["deriv_dir"]],
            config={
                "analysis_type": "one-sample",
                "file_pattern": "**/*_task-nback_stat-effect_statmap.nii.gz",
                "output": {
                    "generate_report": True,
                    "report_filename": "test_report.html",
                },
            },
        )
        
        results = pipeline.run()
        
        report_path = data["output_dir"] / "test_report.html"
        assert report_path.exists()
        
        # Check report content
        content = report_path.read_text()
        assert "StatCraft" in content
        assert "Design Matrix" in content


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_cli_info(self):
        """Test statcraft info command."""
        from click.testing import CliRunner
        from statcraft.cli import info
        
        runner = CliRunner()
        result = runner.invoke(info)
        
        assert result.exit_code == 0
        assert "StatCraft" in result.output
    
    def test_cli_init_config(self, temp_dir):
        """Test statcraft init-config command."""
        from click.testing import CliRunner
        from statcraft.cli import init_config
        
        runner = CliRunner()
        config_path = temp_dir / "test_config.yaml"
        
        result = runner.invoke(init_config, [str(config_path)])
        
        assert result.exit_code == 0
        assert config_path.exists()
        
        # Verify YAML content
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "analysis_type" in config
        assert "inference" in config
