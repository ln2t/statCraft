"""
Design matrix builder for second-level GLM analysis.

This module handles:
- Design matrix generation from participant metadata
- Contrast vector creation
- Automatic contrast naming
- Design matrix validation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nilearn.glm import expression_to_contrast_vector

logger = logging.getLogger(__name__)


class DesignMatrixBuilder:
    """
    Build design matrices and contrasts for second-level GLM.
    
    Parameters
    ----------
    participants : pd.DataFrame
        Participant metadata (from participants.tsv).
    
    Attributes
    ----------
    participants : pd.DataFrame
        Participant metadata.
    design_matrix : pd.DataFrame or None
        Built design matrix.
    contrasts : dict
        Dictionary of contrast name -> contrast vector.
    """
    
    def __init__(self, participants: pd.DataFrame):
        self.participants = participants.copy()
        self.design_matrix: Optional[pd.DataFrame] = None
        self.contrasts: Dict[str, np.ndarray] = {}
    
    def build_design_matrix(
        self,
        columns: List[str],
        add_intercept: bool = True,
        categorical_columns: Optional[List[str]] = None,
        standardize_continuous: bool = True,
    ) -> pd.DataFrame:
        """
        Build a design matrix from participant metadata.
        
        Parameters
        ----------
        columns : list of str
            Column names from participants.tsv to include.
        add_intercept : bool
            Whether to add an intercept column.
        categorical_columns : list of str, optional
            Columns to treat as categorical (will be dummy-coded).
            If None, will auto-detect based on dtype.
        standardize_continuous : bool
            Whether to standardize (z-score) continuous variables.
        
        Returns
        -------
        pd.DataFrame
            Design matrix with rows = participants, columns = regressors.
        """
        design_matrix = pd.DataFrame()
        
        # Validate columns exist
        missing = set(columns) - set(self.participants.columns)
        if missing:
            raise ValueError(f"Columns not found in participants.tsv: {missing}")
        
        # Determine categorical columns
        if categorical_columns is None:
            categorical_columns = []
            for col in columns:
                if self.participants[col].dtype == object or \
                   self.participants[col].nunique() <= 5:
                    categorical_columns.append(col)
        
        # Process each column
        for col in columns:
            if col in categorical_columns:
                # Dummy code categorical variables
                dummies = pd.get_dummies(
                    self.participants[col],
                    prefix=col,
                    drop_first=False,
                )
                design_matrix = pd.concat([design_matrix, dummies], axis=1)
            else:
                # Continuous variable
                values = self.participants[col].astype(float)
                if standardize_continuous:
                    values = (values - values.mean()) / values.std()
                design_matrix[col] = values
        
        # Add intercept
        if add_intercept:
            design_matrix.insert(0, "intercept", 1.0)
        
        self.design_matrix = design_matrix
        logger.info(f"Built design matrix with shape {design_matrix.shape}")
        logger.info(f"Design matrix columns: {list(design_matrix.columns)}")
        
        return design_matrix
    
    def build_group_design_matrix(
        self,
        group_column: str,
        groups: Optional[List[str]] = None,
        add_intercept: bool = False,
    ) -> pd.DataFrame:
        """
        Build a design matrix for group comparisons.
        
        Parameters
        ----------
        group_column : str
            Column name containing group labels.
        groups : list of str, optional
            Specific groups to include. If None, use all unique values.
        add_intercept : bool
            Whether to add an intercept column.
        
        Returns
        -------
        pd.DataFrame
            Design matrix with one column per group.
        """
        if group_column not in self.participants.columns:
            raise ValueError(f"Column '{group_column}' not found in participants.tsv")
        
        if groups is None:
            groups = self.participants[group_column].unique().tolist()
        
        design_matrix = pd.DataFrame()
        
        for group in groups:
            design_matrix[group] = (self.participants[group_column] == group).astype(float)
        
        if add_intercept:
            design_matrix.insert(0, "intercept", 1.0)
        
        self.design_matrix = design_matrix
        logger.info(f"Built group design matrix with groups: {groups}")
        
        return design_matrix
    
    def build_one_sample_design_matrix(
        self,
        n_subjects: int,
    ) -> pd.DataFrame:
        """
        Build a simple design matrix for one-sample t-test.
        
        Parameters
        ----------
        n_subjects : int
            Number of subjects.
        
        Returns
        -------
        pd.DataFrame
            Design matrix with single intercept column.
        """
        design_matrix = pd.DataFrame({"intercept": np.ones(n_subjects)})
        self.design_matrix = design_matrix
        logger.info(f"Built one-sample design matrix for {n_subjects} subjects")
        
        return design_matrix
    
    def build_two_sample_design_matrix(
        self,
        group_labels: List[str],
        group_names: Tuple[str, str] = ("group1", "group2"),
    ) -> pd.DataFrame:
        """
        Build a design matrix for two-sample t-test.
        
        Parameters
        ----------
        group_labels : list of str
            Group label for each subject.
        group_names : tuple of str
            Names for the two groups.
        
        Returns
        -------
        pd.DataFrame
            Design matrix with two columns (one per group).
        """
        unique_groups = list(set(group_labels))
        if len(unique_groups) != 2:
            raise ValueError(f"Expected 2 groups, found {len(unique_groups)}: {unique_groups}")
        
        design_matrix = pd.DataFrame()
        design_matrix[group_names[0]] = [1.0 if g == unique_groups[0] else 0.0 for g in group_labels]
        design_matrix[group_names[1]] = [1.0 if g == unique_groups[1] else 0.0 for g in group_labels]
        
        self.design_matrix = design_matrix
        logger.info(f"Built two-sample design matrix: {unique_groups}")
        
        return design_matrix
    
    def build_paired_design_matrix(
        self,
        n_pairs: int,
    ) -> pd.DataFrame:
        """
        Build a design matrix for paired t-test.
        
        Creates a design with subject-specific intercepts and a condition effect.
        
        Parameters
        ----------
        n_pairs : int
            Number of subject pairs.
        
        Returns
        -------
        pd.DataFrame
            Design matrix for paired test.
        """
        # For paired t-test, we model the difference
        # Simple approach: intercept only (testing against zero)
        design_matrix = pd.DataFrame({"intercept": np.ones(n_pairs)})
        
        self.design_matrix = design_matrix
        logger.info(f"Built paired design matrix for {n_pairs} pairs")
        
        return design_matrix
    
    def add_contrast(
        self,
        contrast_expression: str,
        name: Optional[str] = None,
    ) -> Tuple[str, np.ndarray]:
        """
        Add a contrast from a string expression.
        
        Uses nilearn's expression_to_contrast_vector for parsing.
        
        Parameters
        ----------
        contrast_expression : str
            Contrast expression (e.g., "age", "group1-group2", "patients-controls").
        name : str, optional
            Custom name for the contrast. If None, auto-generated.
        
        Returns
        -------
        tuple of (str, np.ndarray)
            Contrast name and contrast vector.
        
        Examples
        --------
        >>> builder.add_contrast("age")  # Effect of age
        >>> builder.add_contrast("patients-controls")  # Group difference
        >>> builder.add_contrast("0.5*group1 + 0.5*group2")  # Average effect
        """
        if self.design_matrix is None:
            raise ValueError("Must build design matrix before adding contrasts")
        
        # Get column names
        column_names = list(self.design_matrix.columns)
        
        # Use nilearn's expression parser
        try:
            contrast_vector = expression_to_contrast_vector(
                contrast_expression,
                column_names,
            )
        except Exception as e:
            logger.error(f"Failed to parse contrast '{contrast_expression}': {e}")
            raise ValueError(f"Invalid contrast expression: {contrast_expression}") from e
        
        # Generate contrast name if not provided
        if name is None:
            name = self._generate_contrast_name(contrast_expression)
        
        self.contrasts[name] = contrast_vector
        logger.info(f"Added contrast '{name}': {contrast_expression} -> {contrast_vector}")
        
        return name, contrast_vector
    
    def add_contrasts_from_config(
        self,
        contrast_specs: List[Union[str, Dict[str, Any]]],
    ) -> Dict[str, np.ndarray]:
        """
        Add multiple contrasts from configuration.
        
        Parameters
        ----------
        contrast_specs : list
            List of contrast specifications. Each can be:
            - A string expression (e.g., "age", "group1-group2")
            - A dict with "expression" and optional "name" keys
        
        Returns
        -------
        dict
            Dictionary of contrast name -> contrast vector.
        """
        for spec in contrast_specs:
            if isinstance(spec, str):
                self.add_contrast(spec)
            elif isinstance(spec, dict):
                expression = spec.get("expression", spec.get("contrast"))
                name = spec.get("name")
                self.add_contrast(expression, name)
            else:
                raise ValueError(f"Invalid contrast specification: {spec}")
        
        return self.contrasts
    
    def _generate_contrast_name(self, expression: str) -> str:
        """
        Generate a descriptive name from a contrast expression.
        
        Parameters
        ----------
        expression : str
            Contrast expression.
        
        Returns
        -------
        str
            Generated contrast name.
        """
        # Clean up expression
        expr = expression.strip()
        
        # Simple variable name
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", expr):
            return f"effectOf{expr.capitalize()}"
        
        # Subtraction: A - B
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*-\s*([a-zA-Z_][a-zA-Z0-9_]*)$", expr)
        if match:
            return f"{match.group(1)}Versus{match.group(2).capitalize()}"
        
        # Addition: A + B
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\+\s*([a-zA-Z_][a-zA-Z0-9_]*)$", expr)
        if match:
            return f"{match.group(1)}Plus{match.group(2).capitalize()}"
        
        # Complex expression - create a hash-based name
        clean = re.sub(r"[^a-zA-Z0-9]", "", expr)
        return f"contrast_{clean[:20]}"
    
    def get_design_matrix_for_nilearn(self) -> pd.DataFrame:
        """
        Get the design matrix in a format suitable for nilearn.
        
        Returns
        -------
        pd.DataFrame
            Design matrix ready for nilearn's SecondLevelModel.
        """
        if self.design_matrix is None:
            raise ValueError("Design matrix not built yet")
        
        return self.design_matrix.copy()
    
    def get_contrast_vector(self, name: str) -> np.ndarray:
        """
        Get a contrast vector by name.
        
        Parameters
        ----------
        name : str
            Contrast name.
        
        Returns
        -------
        np.ndarray
            Contrast vector.
        """
        if name not in self.contrasts:
            raise KeyError(f"Contrast '{name}' not found. Available: {list(self.contrasts.keys())}")
        return self.contrasts[name]
    
    def summary(self) -> str:
        """
        Get a text summary of the design matrix and contrasts.
        
        Returns
        -------
        str
            Summary text.
        """
        lines = []
        
        if self.design_matrix is not None:
            lines.append("Design Matrix:")
            lines.append(f"  Shape: {self.design_matrix.shape}")
            lines.append(f"  Columns: {list(self.design_matrix.columns)}")
            lines.append("")
        
        if self.contrasts:
            lines.append("Contrasts:")
            for name, vector in self.contrasts.items():
                lines.append(f"  {name}: {vector}")
        
        return "\n".join(lines)


def create_contrast_from_string(
    expression: str,
    design_matrix: pd.DataFrame,
) -> Tuple[str, np.ndarray]:
    """
    Utility function to create a contrast from a string expression.
    
    Parameters
    ----------
    expression : str
        Contrast expression.
    design_matrix : pd.DataFrame
        Design matrix.
    
    Returns
    -------
    tuple of (str, np.ndarray)
        Auto-generated contrast name and contrast vector.
    """
    builder = DesignMatrixBuilder(pd.DataFrame())
    builder.design_matrix = design_matrix
    return builder.add_contrast(expression)
