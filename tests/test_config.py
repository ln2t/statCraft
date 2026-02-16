"""Tests for the configuration module."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from statcraft.config import Config, load_config, create_default_config, DEFAULT_CONFIG


class TestConfig:
    """Tests for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.get("alpha") is None  # Nested under inference
        assert config.get("inference.alpha") == 0.05
        assert config.get("inference.height_threshold") == 0.001
        assert config.get("atlas") == "harvard_oxford"
    
    def test_config_override(self):
        """Test configuration override with kwargs."""
        config = Config(
            analysis_type="one-sample",
            inference={"alpha": 0.01},
        )
        
        assert config.get("analysis_type") == "one-sample"
        assert config.get("inference.alpha") == 0.01
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
analysis_type: two-sample
inference:
  alpha: 0.01
  height_threshold: 0.005
contrasts:
  - "group1 - group2"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = Config(config_file=f.name)
        
        assert config.get("analysis_type") == "two-sample"
        assert config.get("inference.alpha") == 0.01
        assert len(config.get("contrasts")) == 1
        
        Path(f.name).unlink()
    
    def test_config_from_json(self):
        """Test loading configuration from JSON file."""
        json_content = {
            "analysis_type": "paired",
            "inference": {"alpha": 0.1},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            f.flush()
            
            config = Config(config_file=f.name)
        
        assert config.get("analysis_type") == "paired"
        assert config.get("inference.alpha") == 0.1
        
        Path(f.name).unlink()
    
    def test_config_save_yaml(self):
        """Test saving configuration to YAML."""
        config = Config(analysis_type="glm")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.save_to_file(f.name)
        
        # Reload and verify
        with open(f.name) as f2:
            saved = yaml.safe_load(f2)
        
        assert saved["analysis_type"] == "glm"
        
        Path(f.name).unlink()
    
    def test_config_save_json(self):
        """Test saving configuration to JSON."""
        config = Config(analysis_type="one-sample")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save_to_file(f.name)
        
        # Reload and verify
        with open(f.name) as f2:
            saved = json.load(f2)
        
        assert saved["analysis_type"] == "one-sample"
        
        Path(f.name).unlink()
    
    def test_config_validation_valid(self):
        """Test configuration validation with valid config."""
        # Should not raise
        config = Config(analysis_type="glm")
        config.validate()
    
    def test_config_validation_invalid_alpha(self):
        """Test configuration validation with invalid alpha."""
        with pytest.raises(ValueError, match="alpha"):
            Config(inference={"alpha": 1.5})
    
    def test_config_validation_invalid_analysis_type(self):
        """Test configuration validation with invalid analysis type."""
        with pytest.raises(ValueError, match="analysis_type"):
            Config(analysis_type="invalid_type")
    
    def test_config_validation_paired_missing_params(self):
        """Test configuration validation for paired analysis."""
        with pytest.raises(ValueError, match="pair_by"):
            Config(analysis_type="paired")
    
    def test_config_get_nested(self):
        """Test getting nested configuration values."""
        config = Config()
        
        assert config.get("inference.alpha") == 0.05
        assert config.get("inference.corrections") is not None
        assert config.get("nonexistent.key", "default") == "default"
    
    def test_config_set_nested(self):
        """Test setting nested configuration values."""
        config = Config()
        
        config.set("inference.alpha", 0.1)
        assert config.get("inference.alpha") == 0.1
        
        config.set("new.nested.value", "test")
        assert config.get("new.nested.value") == "test"
    
    def test_config_dict_access(self):
        """Test dictionary-style access."""
        config = Config()
        
        assert config["analysis_type"] == "glm"
        
        config["analysis_type"] = "one-sample"
        assert config["analysis_type"] == "one-sample"
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(analysis_type="two-sample")
        
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert d["analysis_type"] == "two-sample"
    
    def test_config_summary(self):
        """Test configuration summary."""
        config = Config(
            analysis_type="glm",
            contrasts=["age", "group1 - group2"],
        )
        
        summary = config.summary()
        
        assert "glm" in summary
        assert "Contrasts" in summary


class TestConfigHelpers:
    """Tests for configuration helper functions."""
    
    def test_load_config(self):
        """Test load_config helper function."""
        config = load_config(analysis_type="one-sample")
        
        assert isinstance(config, Config)
        assert config.get("analysis_type") == "one-sample"
    
    def test_create_default_config(self):
        """Test create_default_config helper function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = create_default_config(f.name)
        
        assert path.exists()
        
        # Verify it's valid YAML
        with open(path) as f2:
            saved = yaml.safe_load(f2)
        
        assert "analysis_type" in saved
        
        path.unlink()
