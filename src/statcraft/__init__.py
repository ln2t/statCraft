"""
StatCraft: Second-level neuroimaging analysis tool.

A pip-installable Python tool for second-level neuroimaging analysis,
supporting group-level comparisons, method comparisons, and statistical
inference on brain images (e.g., fMRI, PET).
"""

__version__ = "0.1.0"
__author__ = "StatCraft Contributors"

from statcraft.core.data_loader import DataLoader
from statcraft.core.design_matrix import DesignMatrixBuilder
from statcraft.core.glm import SecondLevelGLM
from statcraft.core.inference import StatisticalInference
from statcraft.core.annotation import ClusterAnnotator
from statcraft.core.report import ReportGenerator
from statcraft.pipeline import StatCraftPipeline

__all__ = [
    "DataLoader",
    "DesignMatrixBuilder",
    "SecondLevelGLM",
    "StatisticalInference",
    "ClusterAnnotator",
    "ReportGenerator",
    "StatCraftPipeline",
    "__version__",
]
