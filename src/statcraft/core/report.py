"""
Report generator module for HTML reports.

This module handles:
- HTML report generation with Jinja2
- Visualization of statistical maps
- Design matrix plots
- Cluster table formatting
"""

import base64
import io
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, PackageLoader, select_autoescape
from nilearn import plotting, datasets
from nilearn.plotting import cm as nilearn_cm
from nilearn.image import mean_img, resample_to_img

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate HTML reports for second-level analysis results.
    
    Parameters
    ----------
    title : str
        Report title.
    output_dir : str or Path
        Directory for saving the report.
    
    Attributes
    ----------
    title : str
        Report title.
    output_dir : Path
        Output directory.
    sections : list
        Report sections.
    """
    
    def __init__(
        self,
        title: str = "StatCraft Analysis Report",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.title = title
        self.output_dir = Path(output_dir) if output_dir else None
        self.sections: List[Dict[str, Any]] = []
        self._figures: Dict[str, str] = {}  # Base64-encoded figures
        self._latex_tables: Dict[str, str] = {}  # LaTeX table code
        self._tsv_data: Dict[str, str] = {}  # TSV table data for download

        # Load MNI152 template for axial slice backgrounds
        try:
            self.mni_template = datasets.load_mni152_template()
            logger.debug("Loaded MNI152 template for axial slice backgrounds")
        except Exception as e:
            self.mni_template = None
            logger.debug(f"Could not load MNI152 template: {e}")

        # Initialize Jinja2 environment
        try:
            self.env = Environment(
                loader=PackageLoader("statcraft", "templates"),
                autoescape=select_autoescape(["html", "xml"]),
            )
        except Exception:
            # Fallback: use simple string templates
            self.env = None
            logger.warning("Could not load Jinja2 templates, using basic HTML")
    
    def add_section(
        self,
        title: str,
        content: str,
        section_type: str = "text",
        level: int = 1,
        caption: Optional[str] = None,
    ) -> None:
        """
        Add a section to the report.
        
        Parameters
        ----------
        title : str
            Section title.
        content : str
            Section content (HTML string, figure key, or table).
        section_type : str
            Type of section: "text", "figure", "table".
        level : int
            Hierarchical level: 1=main section, 2=subsection, 3=sub-subsection.
        caption : str, optional
            Detailed caption or description for the section (especially for figures).
        """
        self.sections.append({
            "title": title,
            "content": content,
            "type": section_type,
            "level": level,
            "caption": caption,
        })
    
    def add_data_files_section(
        self,
        all_images: List[Dict],
        valid_images: List[Dict],
        invalid_images: List[Dict],
        masks: Optional[Union[List[str], Dict[str, str]]] = None,
        scaling_rois: Optional[Union[List[str], Dict[str, str]]] = None,
    ) -> None:
        """
        Add section showing discovered and validated image files and supporting data.

        Parameters
        ----------
        all_images : list of dict
            All discovered image files.
        valid_images : list of dict
            Valid images after validation.
        invalid_images : list of dict
            Invalid images excluded from analysis.
        masks : list of str or dict, optional
            Mask files used in analysis. Can be a list of unique paths or a dict 
            mapping image paths to their corresponding mask paths.
        scaling_rois : list of str or dict, optional
            Scaling ROI files used in analysis. Can be a list of unique paths or a dict
            mapping image paths to their corresponding scaling ROI paths.
        """
        from pathlib import Path
        
        html = f"""
        <h3>Data Discovery</h3>
        <p><strong>Total files discovered:</strong> {len(all_images)}</p>
        <p><strong>Valid files for analysis:</strong> {len(valid_images)}</p>
        <p><strong>Excluded files:</strong> {len(invalid_images)}</p>

        <h3>Valid Image Files</h3>
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">
        <table class="table table-sm table-striped">
            <thead>
                <tr>
                    <th style="width: 50px;">#</th>
                    <th>File Path</th>
                </tr>
            </thead>
            <tbody>
        """

        for i, img in enumerate(valid_images, 1):
            path_str = str(img['path'])
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><code>{path_str}</code></td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        </div>
        """

        # Add masks section if provided
        if masks:
            if isinstance(masks, dict):
                # Show image-to-mask mapping
                html += """
                <h3>Mask Files (Image-to-Mask Mapping)</h3>
                <p style="font-size: 0.9em; color: #666;">Each image was processed using its corresponding mask file for z-scoring.</p>
                <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">
                <table class="table table-sm table-striped">
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            <th style="width: 50%;">Image File</th>
                            <th style="width: 50%;">Mask File</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for i, (img_path, mask_path) in enumerate(sorted(masks.items()), 1):
                    img_name = Path(img_path).name
                    mask_name = Path(mask_path).name
                    html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><code style="font-size: 0.85em;">{img_name}</code></td>
                            <td><code style="font-size: 0.85em;">{mask_name}</code></td>
                        </tr>
                    """
                html += """
                    </tbody>
                </table>
                </div>
                """
            else:
                # Show unique masks (legacy behavior)
                html += """
                <h3>Mask Files</h3>
                <div style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">
                <table class="table table-sm table-striped">
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            <th>File Path</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for i, mask in enumerate(set(masks), 1):
                    html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><code>{mask}</code></td>
                        </tr>
                    """
                html += """
                    </tbody>
                </table>
                </div>
                """

        # Add scaling ROIs section if provided
        if scaling_rois:
            if isinstance(scaling_rois, dict):
                # Show image-to-scaling-ROI mapping
                html += """
                <h3>Scaling ROI Files (Image-to-Mask Mapping)</h3>
                <p style="font-size: 0.9em; color: #666;">Each image was scaled by dividing by the mean value within its corresponding ROI mask.</p>
                <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">
                <table class="table table-sm table-striped">
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            <th style="width: 50%;">Image File</th>
                            <th style="width: 50%;">Scaling ROI Mask</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for i, (img_path, roi_path) in enumerate(sorted(scaling_rois.items()), 1):
                    img_name = Path(img_path).name
                    roi_name = Path(roi_path).name
                    html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><code style="font-size: 0.85em;">{img_name}</code></td>
                            <td><code style="font-size: 0.85em;">{roi_name}</code></td>
                        </tr>
                    """
                html += """
                    </tbody>
                </table>
                </div>
                """
            else:
                # Show unique ROIs (legacy behavior)
                html += """
                <h3>Scaling ROI Files</h3>
                <div style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">
                <table class="table table-sm table-striped">
                    <thead>
                        <tr>
                            <th style="width: 50px;">#</th>
                            <th>File Path</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for i, roi in enumerate(set(scaling_rois), 1):
                    html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><code>{roi}</code></td>
                        </tr>
                    """
                html += """
                    </tbody>
                </table>
            </div>
            """

        if invalid_images:
            html += """
            <h3>Excluded Files</h3>
            <div style="max-height: 300px; overflow-y: auto; border: 1px solid #ffcccc; padding: 10px; background-color: #fff5f5;">
            <table class="table table-sm table-striped">
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>File Path</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
            """

            for i, img in enumerate(invalid_images, 1):
                path_str = str(img['path'])
                reason = img.get('reason', 'Unknown')
                html += f"""
                    <tr>
                        <td>{i}</td>
                        <td><code>{path_str}</code></td>
                        <td><span style="color: #d9534f;">{reason}</span></td>
                    </tr>
                """

            html += """
                </tbody>
            </table>
            </div>
            """

        self.add_section("Data Files", html, "text", level=2)

    def add_methodology_section(
        self,
        design_matrix: pd.DataFrame,
        contrasts: Dict[str, np.ndarray],
        analysis_type: str,
        n_subjects: int,
        correction_methods: List[str],
        alpha_corrected: float = 0.05,
        alpha_uncorrected: float = 0.001,
        smoothing_fwhm: Optional[float] = None,
        zscore: bool = False,
        scaling: Optional[str] = None,
        n_contrasts: int = 0,
        paired_info: Optional[Dict] = None,
    ) -> None:
        """
        Add methodology description section with processing parameters.

        Parameters
        ----------
        design_matrix : pd.DataFrame
            Design matrix used in analysis.
        contrasts : dict
            Dictionary of contrast names to vectors.
        analysis_type : str
            Type of analysis performed.
        n_subjects : int
            Number of subjects.
        correction_methods : list
            List of correction methods used.
        alpha_corrected : float
            Significance level for corrected analyses (FDR, Bonferroni, permutation).
        alpha_uncorrected : float
            Significance level for uncorrected analysis (cluster-forming threshold).
        smoothing_fwhm : float, optional
            Smoothing kernel FWHM in mm.
        zscore : bool
            Whether z-scoring was applied.
        scaling : str, optional
            Scaling method applied.
        n_contrasts : int
            Number of contrasts tested.
        paired_info : dict, optional
            Pairing information for paired analyses with keys:
            - 'pairs': list of pair display strings
            - 'sample1_name': name of first sample
            - 'sample2_name': name of second sample
            - 'pair_by': pairing entity name
        """
        # Create methodology text
        html = f"""
        <h3>Analysis Overview</h3>
        <ul>
            <li><strong>Analysis Type:</strong> {analysis_type}</li>
            <li><strong>Number of Subjects:</strong> {n_subjects}</li>
            <li><strong>Number of Contrasts:</strong> {n_contrasts}</li>
            <li><strong>Correction Methods:</strong> {', '.join(correction_methods)}</li>
        </ul>

        <h3>Statistical Thresholds</h3>
        <p><em>Note: Two different significance levels (α) are used in neuroimaging analysis:</em></p>
        <ul>
            <li><strong>α<sub>uncorrected</sub> (cluster-forming threshold):</strong> {alpha_uncorrected}
                <br><small style="color: #666;">Significance level for uncorrected analysis and initial cluster detection</small></li>
            <li><strong>α<sub>corrected</sub>:</strong> {alpha_corrected}
                <br><small style="color: #666;">Significance level for FDR, Bonferroni, and permutation-corrected analyses</small></li>
        </ul>"""

        # Add paired file information if available
        if paired_info and paired_info.get('pairs'):
            html += f"""
        <h3>Paired File Mapping</h3>
        <p><strong>Pairing Entity:</strong> {paired_info['pair_by']}</p>
        <p><strong>Samples:</strong> {paired_info['sample1_name']} vs {paired_info['sample2_name']}</p>
        <div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; margin: 10px 0;">
            <table class="table table-sm table-striped">
                <thead>
                    <tr>
                        <th>Pairing</th>
                        <th>{paired_info['sample1_name']}</th>
                        <th>{paired_info['sample2_name']}</th>
                    </tr>
                </thead>
                <tbody>
        """

            # Parse pair information and create compact table
            for pair_str in paired_info['pairs']:
                lines = pair_str.strip().split('\n')
                pair_id = lines[0].split(':')[0]  # e.g., "sub=001"
                sample1_file = lines[1].split(':')[1].strip() if len(lines) > 1 else ""
                sample2_file = lines[2].split(':')[1].strip() if len(lines) > 2 else ""

                html += f"""
                    <tr>
                        <td><code>{pair_id}</code></td>
                        <td><code style="font-size: 0.85em;">{sample1_file}</code></td>
                        <td><code style="font-size: 0.85em;">{sample2_file}</code></td>
                    </tr>
                """

            html += """
                </tbody>
            </table>
        </div>
        """

        html += f"""
        <h3>Preprocessing Parameters</h3>
        <ul>
            <li><strong>Smoothing FWHM:</strong> {smoothing_fwhm if smoothing_fwhm else 'None'} mm</li>
            <li><strong>Z-scoring:</strong> {'Yes' if zscore else 'No'}</li>
            <li><strong>Scaling:</strong> {scaling if scaling else 'None'}</li>
        </ul>

        <h3>Design Matrix</h3>
        <p>The design matrix has {design_matrix.shape[0]} rows (observations) and
        {design_matrix.shape[1]} columns (regressors).</p>
        <p><strong>Regressors:</strong> {', '.join(design_matrix.columns)}</p>

        <h3>Contrasts</h3>
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Contrast Name</th>
                    <th>Contrast Vector</th>
                </tr>
            </thead>
            <tbody>
        """

        for name, vector in contrasts.items():
            vector_str = ', '.join([f"{v:.2f}" for v in vector])
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td>[{vector_str}]</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        self.add_section("Methodology", html, "text", level=2)

        # Add design matrix plot
        fig_key = self._plot_design_matrix(design_matrix)
        self.add_section("Design Matrix Visualization", fig_key, "figure", level=2)
        
    def add_activation_maps(
        self,
        stat_map: nib.Nifti1Image,
        thresholded_maps: Dict[str, nib.Nifti1Image],
        contrast_name: str,
    ) -> None:
        """
        Add activation map visualizations (glass brain + axial slices).

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Unthresholded statistical map.
        thresholded_maps : dict
            Dictionary of correction method -> thresholded map.
        contrast_name : str
            Name of the contrast.
        """
        # Unthresholded map
        if self._is_map_empty(stat_map):
            self.add_section(
                f"Contrast: {contrast_name} - Unthresholded",
                "<div class='empty-content'>No significant results to display for this section.</div>",
                "text",
            )
        else:
            # Glass brain view - extract color scale for consistency
            # Use threshold=0 for truly unthresholded visualization
            fig_key, vmin, vmax = self._plot_glass_brain(
                stat_map,
                title=f"{contrast_name} (Unthresholded) - Glass Brain",
                threshold=0,  # No threshold for unthresholded maps
            )
            unthresh_caption = (
                "Unthresholded z-score map showing all voxels (positive and negative effects). "
                "Color intensity represents effect size; no statistical threshold applied. "
                "All data are displayed to show the full spatial extent of effects."
            )
            self.add_section(
                f"Contrast: {contrast_name} - Unthresholded (Glass Brain)",
                fig_key,
                "figure",
                caption=unthresh_caption,
            )

            # Axial slices - use same color scale as glass brain, no threshold
            fig_key = self._plot_axial_slices(
                stat_map,
                title=f"{contrast_name} (Unthresholded) - Axial Slices",
                threshold=0,  # No threshold for unthresholded maps
                vmin=vmin,
                vmax=vmax,
            )
            self.add_section(
                f"Contrast: {contrast_name} - Unthresholded (Axial Slices)",
                fig_key,
                "figure",
                caption=unthresh_caption,
            )

        # Thresholded maps
        for correction, thresh_map in thresholded_maps.items():
            if self._is_map_empty(thresh_map):
                self.add_section(
                    f"Contrast: {contrast_name} - {correction.upper()} Corrected",
                    "<div class='empty-content'>No significant results to display for this section.</div>",
                    "text",
                )
            else:
                # Glass brain view - extract color scale for consistency
                fig_key, vmin, vmax = self._plot_glass_brain(
                    thresh_map,
                    title=f"{contrast_name} ({correction.upper()}) - Glass Brain",
                    threshold=1.96,
                )
                
                # Create informative caption based on correction method
                if correction == "uncorrected":
                    thresh_caption = (
                        f"Thresholded z-score map showing voxels surviving uncorrected threshold (p < 0.001). "
                        "Only voxels with |z| > 1.96 are displayed. "
                        "No correction for multiple comparisons applied; results may include false positives."
                    )
                elif correction == "fdr":
                    thresh_caption = (
                        f"Thresholded z-score map showing voxels surviving FDR correction (q < 0.05). "
                        "False Discovery Rate controls the expected proportion of false positives among suprathreshold voxels. "
                        "Only statistically significant voxels after correction are displayed."
                    )
                elif correction == "bonferroni":
                    thresh_caption = (
                        f"Thresholded z-score map showing voxels surviving Bonferroni correction (p < 0.05, family-wise). "
                        "Controls family-wise error rate with very conservative threshold. "
                        "Only voxels passing stringent correction for multiple comparisons are shown."
                    )
                elif correction == "perm":
                    thresh_caption = (
                        f"Thresholded z-score map showing voxels surviving permutation-based correction (p < 0.05, FWER). "
                        "Non-parametric permutation testing controls family-wise error rate. "
                        "Only cluster-level significant results are displayed."
                    )
                else:
                    thresh_caption = (
                        f"Thresholded z-score map showing voxels surviving {correction.upper()} correction. "
                        "Only statistically significant voxels are displayed."
                    )
                
                self.add_section(
                    f"Contrast: {contrast_name} - {correction.upper()} (Glass Brain)",
                    fig_key,
                    "figure",
                    caption=thresh_caption,
                )

                # Axial slices - use same color scale as glass brain
                fig_key = self._plot_axial_slices(
                    thresh_map,
                    title=f"{contrast_name} ({correction.upper()}) - Axial Slices",
                    threshold=1.96,  # Display threshold for thresholded maps
                    vmin=vmin,
                    vmax=vmax,
                )
                self.add_section(
                    f"Contrast: {contrast_name} - {correction.upper()} (Axial Slices)",
                    fig_key,
                    "figure",
                    caption=thresh_caption,
                )
    
    def add_glass_brain(
        self,
        stat_map: nib.Nifti1Image,
        contrast_name: str,
        threshold: float = 3.0,
    ) -> None:
        """
        Add glass brain visualization.
        
        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map.
        contrast_name : str
            Name of the contrast.
        threshold : float
            Display threshold.
        """
        fig_key = self._plot_glass_brain(
            stat_map,
            title=f"{contrast_name} - Glass Brain",
            threshold=threshold,
        )
        self.add_section(
            f"Contrast: {contrast_name} - Glass Brain View",
            fig_key,
            "figure",
        )

    def _generate_latex_table(
        self,
        df: pd.DataFrame,
        table_key: str,
        caption: str = "",
        label: str = "",
    ) -> str:
        """
        Generate LaTeX code from a DataFrame using booktabs format.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert.
        table_key : str
            Unique key for storing the LaTeX code.
        caption : str
            Table caption.
        label : str
            LaTeX label reference.

        Returns
        -------
        str
            LaTeX code for the table.
        """
        # Convert DataFrame to LaTeX using booktabs format
        latex_code = df.to_latex(
            index=False,
            float_format=lambda x: f"{x:.3f}",
            escape=True,
            caption=caption if caption else None,
            label=label if label else None,
            column_format="l" + "c" * (len(df.columns) - 1),
        )

        # Modify to use booktabs commands
        if latex_code:
            latex_code = latex_code.replace("\\hline", "")
            if "\\begin{table}" in latex_code:
                latex_code = latex_code.replace(
                    "\\begin{tabular}",
                    "\\begin{tabular}"
                )
                latex_code = latex_code.replace(
                    "\\toprule" if "\\toprule" in latex_code else "\\hline",
                    "\\toprule"
                )
                if "\\bottomrule" not in latex_code:
                    # Add booktabs-style bottom rule
                    lines = latex_code.split("\n")
                    for i, line in enumerate(lines):
                        if "\\end{tabular}" in line:
                            lines.insert(i, "\\bottomrule")
                            break
                    latex_code = "\n".join(lines)

        # Store for later download
        self._latex_tables[table_key] = latex_code

        return latex_code

    def _store_tsv_data(self, tsv_key: str, tsv_data: str) -> None:
        """
        Store TSV data for later download.

        Parameters
        ----------
        tsv_key : str
            Unique key for this TSV data.
        tsv_data : str
            TSV-formatted string data.
        """
        self._tsv_data[tsv_key] = tsv_data

    def add_cluster_table(
        self,
        cluster_table: pd.DataFrame,
        contrast_name: str,
        correction: str,
    ) -> None:
        """
        Add a cluster table to the report.

        Parameters
        ----------
        cluster_table : pd.DataFrame
            Cluster table.
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method.
        """
        if len(cluster_table) == 0:
            html = f"<p>No significant clusters found for {contrast_name} ({correction}).</p>"
        else:
            html = cluster_table.to_html(
                classes="table table-striped table-hover",
                index=False,
                float_format="%.3f",
            )

            # Generate LaTeX version
            table_key = f"cluster_{contrast_name.lower().replace(' ', '_')}_{correction.lower()}"
            latex_caption = f"Cluster table for {contrast_name} ({correction.upper()})"
            self._generate_latex_table(
                cluster_table,
                table_key,
                caption=latex_caption,
                label=table_key,
            )

            # Store TSV data for download
            tsv_key = f"tsv_{table_key}"
            tsv_data = cluster_table.to_csv(sep='\t', index=False)
            self._store_tsv_data(tsv_key, tsv_data)

            # Add download buttons
            # Escape contrast name for JavaScript (replace single quotes with &#39;)
            escaped_name = contrast_name.replace("'", "&#39;")
            html += f"""
            <div style="margin-top: 1rem;">
                <button class="btn-download-tsv" onclick="downloadTSV('{tsv_key}', '{escaped_name}_clusters_{correction}')">
                    ⬇ Download TSV
                </button>
                <button class="btn-download-latex" onclick="downloadLatexTable('{table_key}', '{escaped_name}_clusters_{correction}')">
                    ⬇ Download LaTeX
                </button>
            </div>
            """

        self.add_section(
            f"Cluster Table",
            html,
            "table",
            level=3,
        )

    def add_enhanced_cluster_table(
        self,
        enhanced_table: pd.DataFrame,
        contrast_name: str,
        correction: str,
    ) -> None:
        """
        Add an enhanced cluster table with region percentages to the report.

        Parameters
        ----------
        enhanced_table : pd.DataFrame
            Enhanced cluster table with region overlap information.
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method.
        """
        if len(enhanced_table) == 0:
            html = f"<p>No significant clusters found for {contrast_name} ({correction}).</p>"
        else:
            # Style the table with better formatting for the Regions column
            html = enhanced_table.to_html(
                classes="table table-striped table-hover table-sm",
                index=False,
                float_format="%.3f",
                escape=False,  # Allow HTML in cells if needed
            )

            # Generate LaTeX version
            table_key = f"enhanced_{contrast_name.lower().replace(' ', '_')}_{correction.lower()}"
            latex_caption = f"Enhanced cluster table for {contrast_name} ({correction.upper()})"
            self._generate_latex_table(
                enhanced_table,
                table_key,
                caption=latex_caption,
                label=table_key,
            )

            # Store TSV data for download
            tsv_key = f"tsv_{table_key}"
            tsv_data = enhanced_table.to_csv(sep='\t', index=False)
            self._store_tsv_data(tsv_key, tsv_data)

            # Add download buttons
            escaped_name = contrast_name.replace("'", "&#39;")
            html += f"""
            <div style="margin-top: 1rem;">
                <button class="btn-download-tsv" onclick="downloadTSV('{tsv_key}', '{escaped_name}_enhanced_clusters_{correction}')">
                    ⬇ Download TSV
                </button>
                <button class="btn-download-latex" onclick="downloadLatexTable('{table_key}', '{escaped_name}_enhanced_clusters_{correction}')">
                    ⬇ Download LaTeX
                </button>
            </div>
            """

        self.add_section(
            f"Enhanced Cluster Table",
            html,
            "table",
            level=3,
        )
    def add_cortical_summary(
        self,
        cortical_summary: pd.DataFrame,
        contrast_name: str,
        correction: str,
    ) -> None:
        """
        Add cortical/non-cortical summary statistics table to the report.

        Parameters
        ----------
        cortical_summary : pd.DataFrame
            Summary of cortical/non-cortical positive/negative clusters.
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method.
        """
        if len(cortical_summary) == 0:
            html = f"<p>No cortical/non-cortical summary available for {contrast_name} ({correction}).</p>"
        else:
            html = cortical_summary.to_html(
                classes="table table-bordered table-hover",
                index=False,
                float_format="%.3f",
            )

            # Generate LaTeX version
            table_key = f"cortical_{contrast_name.lower().replace(' ', '_')}_{correction.lower()}"
            latex_caption = f"Cortical/Non-Cortical summary for {contrast_name} ({correction.upper()})"
            self._generate_latex_table(
                cortical_summary,
                table_key,
                caption=latex_caption,
                label=table_key,
            )

            # Store TSV data for download
            tsv_key = f"tsv_{table_key}"
            tsv_data = cortical_summary.to_csv(sep='\t', index=False)
            self._store_tsv_data(tsv_key, tsv_data)

            # Add download buttons
            escaped_name = contrast_name.replace("'", "&#39;")
            html += f"""
            <div style="margin-top: 1rem;">
                <button class="btn-download-tsv" onclick="downloadTSV('{tsv_key}', '{escaped_name}_cortical_summary_{correction}')">
                    ⬇ Download TSV
                </button>
                <button class="btn-download-latex" onclick="downloadLatexTable('{table_key}', '{escaped_name}_cortical_summary_{correction}')">
                    ⬇ Download LaTeX
                </button>
            </div>
            """

        self.add_section(
            f"Cortical/Non-Cortical Summary",
            html,
            "table",
            level=3,
        )

    def add_technical_section(
        self,
        command_line: str,
        config_filename: Optional[str] = None,
        config_content: Optional[str] = None,
    ) -> None:
        """
        Add technical reproduction section with command line and embedded config.

        Parameters
        ----------
        command_line : str
            Command line used to reproduce the analysis.
        config_filename : str, optional
            Path to the configuration file (for filename only).
        config_content : str, optional
            YAML content of the configuration file to embed in HTML.
        """
        html = """
        <h3>Reproduction Information</h3>
        <p>To reproduce this analysis with the exact same configuration, use the following command:</p>
        <pre><code>"""
        html += command_line
        html += """</code></pre>
        """

        if config_filename:
            config_name = Path(config_filename).name
            html += f"""
        <div style="margin-top: 1.5rem;">
            <p><strong>Configuration File:</strong> <code>{config_name}</code></p>
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <button id="config-reveal-btn" style="
                    background-color: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.95rem;
                    transition: background-color 0.3s ease;
                " onclick="toggleConfigContent()">
                    Show Configuration
                </button>
                <button id="config-download-btn" style="
                    background-color: var(--success-border);
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.95rem;
                    text-decoration: none;
                    display: inline-block;
                    transition: background-color 0.3s ease;
                " onclick="downloadConfigFile()">
                    Download YAML
                </button>
            </div>
        </div>
        """

            if config_content:
                # Escape HTML special characters in config content
                escaped_content = (
                    config_content.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )
                html += f"""
        <div id="config-content" style="
            display: none;
            background-color: #f8f9fa;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
            margin-top: 1rem;
            max-height: 500px;
            overflow-y: auto;
        ">
            <pre style="
                margin: 0;
                font-family: 'Courier New', monospace;
                font-size: 0.85rem;
                line-height: 1.4;
            "><code id="config-text">{escaped_content}</code></pre>
        </div>
        """
                # Store config content for download via data attribute
                html += f"""
        <!-- Hidden storage of config content for download -->
        <div id="config-data" style="display: none;" data-config='{config_content.replace("'", "&apos;")}'></div>
        """
                # Store config content for download
                self._config_content = config_content
                self._config_filename = config_name

        self.add_section("Technical Details", html, "text", level=2)

    def add_citation_section(self) -> None:
        """
        Add citation boilerplate section with software reference.
        """
        html = """
        <h3>How to Cite</h3>
        <p>If you use StatCraft in your research, please cite:</p>
        <div style="background-color: #f8f9fa; border-left: 4px solid var(--primary-color); padding: 1rem; margin: 1rem 0;">
                <p style="margin: 0; font-family: 'Courier New', monospace; font-size: 0.9rem;">
                StatCraft: Second-Level Neuroimaging Analysis Tool<br>
                Available at: <a href="https://github.com/ln2t/StatCraft" target="_blank">https://github.com/ln2t/StatCraft</a>
            </p>
        </div>
        
        <h3>Dependencies</h3>
        <p>StatCraft builds upon several excellent open-source neuroimaging packages:</p>
        <ul>
            <li><strong>Nilearn</strong>: Statistical learning for neuroimaging - <a href="https://nilearn.github.io/" target="_blank">https://nilearn.github.io/</a></li>
            <li><strong>Nibabel</strong>: Read/write access to neuroimaging file formats - <a href="https://nipy.org/nibabel/" target="_blank">https://nipy.org/nibabel/</a></li>
            <li><strong>NumPy</strong> and <strong>SciPy</strong>: Fundamental packages for scientific computing</li>
        </ul>
        
        <h3>Source Code</h3>
        <p>The complete source code is available at: <a href="https://github.com/ln2t/StatCraft" target="_blank">https://github.com/ln2t/StatCraft</a></p>
        """
        self.add_section("Citation", html, "text", level=1)

    def add_permutation_null_distribution(
        self,
        h0_distribution: np.ndarray,
        alpha: float,
        contrast_name: str,
        n_perm: int,
    ) -> None:
        """
        Add permutation null distribution visualization.

        Creates a histogram of the null distribution of maximum statistics
        with the alpha-level threshold marked in red.

        Parameters
        ----------
        h0_distribution : np.ndarray
            Null distribution of maximum t-statistics from permutations.
        alpha : float
            Significance level.
        contrast_name : str
            Name of the contrast.
        n_perm : int
            Number of permutations performed.
        """
        if h0_distribution is None or len(h0_distribution) == 0:
            logger.warning("No null distribution available for visualization")
            return

        # Ensure h0_distribution is 1D (flatten if needed)
        h0_dist = np.asarray(h0_distribution)
        if h0_dist.ndim > 1:
            logger.info(f"Flattening h0_distribution from shape {h0_dist.shape}")
            h0_dist = h0_dist.ravel()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram of null distribution
        ax.hist(h0_dist, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black', label='Null Distribution')

        # Calculate and plot threshold
        threshold = np.percentile(h0_dist, (1 - alpha) * 100)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'α = {alpha} threshold ({threshold:.3f})')

        # Add labels and title
        ax.set_xlabel('Maximum t-statistic', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Permutation Null Distribution\n{contrast_name} ({n_perm} permutations)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"N permutations: {n_perm}\n"
        stats_text += f"Mean: {np.mean(h0_dist):.3f}\n"
        stats_text += f"SD: {np.std(h0_dist):.3f}\n"
        stats_text += f"Threshold (α={alpha}): {threshold:.3f}"
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Encode to base64
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            logger.info(f"Successfully created null distribution plot (base64 length: {len(img_base64)})")

            # Store figure with a unique key
            fig_key = f"null_dist_{contrast_name}"
            self._figures[fig_key] = img_base64

            # Add the figure section (content should be the figure key, not HTML)
            self.add_section(
                f"Permutation Null Distribution: {contrast_name}",
                fig_key,
                "figure",
            )

            logger.info(f"Added permutation null distribution section for {contrast_name}")

        except Exception as e:
            logger.error(f"Failed to create null distribution plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Add error message to report
            html = f'<p style="color: red;">Error generating null distribution plot: {e}</p>'
            self.add_section(
                f"Permutation Null Distribution: {contrast_name}",
                html,
                "text",
            )

    def _plot_design_matrix(self, design_matrix: pd.DataFrame) -> str:
        """Plot design matrix and return base64-encoded figure."""
        fig, ax = plt.subplots(figsize=(10, max(4, len(design_matrix) * 0.2)))
        
        # Ensure all columns are numeric for visualization
        design_matrix_numeric = design_matrix.astype(float)
        
        # Plot design matrix
        sns.heatmap(
            design_matrix_numeric,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            cbar=True,
            yticklabels=False,
        )
        ax.set_title("Design Matrix")
        ax.set_xlabel("Regressors")
        ax.set_ylabel("Observations")
        
        plt.tight_layout()
        
        # Convert to base64
        fig_key = self._fig_to_base64(fig)
        plt.close(fig)
        
        return fig_key
    
    def _is_map_empty(self, stat_map: nib.Nifti1Image) -> bool:
        """
        Check if a statistical map is empty (all zeros or all NaN).

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map to check.

        Returns
        -------
        bool
            True if map is empty, False otherwise.
        """
        if stat_map is None:
            return True

        try:
            data = stat_map.get_fdata()
            # Check if all values are zero or NaN
            valid_data = data[~np.isnan(data)]
            return np.all(valid_data == 0) or len(valid_data) == 0
        except Exception as e:
            logger.warning(f"Error checking if map is empty: {e}")
            return False

    def _plot_axial_slices(
        self,
        stat_map: nib.Nifti1Image,
        title: str = "",
        threshold: float = 1.96,
        n_slices: int = 12,
        n_cols: int = 3,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> str:
        """
        Plot grid of axial slices from a statistical map using matplotlib.

        Creates a grid layout of axial slices spanning multiple rows for better visibility.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map to plot.
        title : str
            Title for the plot.
        threshold : float
            Display threshold.
        n_slices : int
            Number of axial slices to display.
        n_cols : int
            Number of columns in the grid (default 3).
        vmin : float, optional
            Minimum value for color scale. If None, computed from data.
        vmax : float, optional
            Maximum value for color scale. If None, computed from data.

        Returns
        -------
        str
            Base64-encoded figure key.
        """
        try:
            data = stat_map.get_fdata()

            # Get z-indices for slices distributed across the brain
            # Span from 10% to 90% of z-range to avoid uninteresting edge slices
            z_shape = data.shape[2]
            z_min = int(z_shape * 0.10)  # Start at 10%
            z_max = int(z_shape * 0.80)  # End at 80%
            z_indices = np.linspace(z_min, z_max, n_slices, dtype=int)

            # Create grid
            n_rows = (n_slices + n_cols - 1) // n_cols  # Ceiling division
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

            # Flatten axes for easier iteration
            if n_slices > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            # Compute vmin/vmax if not provided
            if vmin is None or vmax is None:
                vmin, vmax = self._compute_color_scale(stat_map, threshold=threshold)

            # Plot each slice
            im = None
            for idx, z_idx in enumerate(z_indices):
                ax = axes[idx]
                slice_data = data[:, :, z_idx]

                # Display MNI template as background if available
                if self.mni_template is not None:
                    try:
                        # Resample template to match stat_map if needed
                        template_resampled = resample_to_img(self.mni_template, stat_map)
                        template_data = template_resampled.get_fdata()
                        template_slice = template_data[:, :, z_idx]

                        # Plot template in grayscale as background
                        ax.imshow(
                            template_slice.T,
                            cmap="gray",
                            vmin=np.percentile(template_slice[template_slice > 0], 5),
                            vmax=np.percentile(template_slice[template_slice > 0], 95),
                            origin="lower",
                            interpolation="bilinear",
                            alpha=1,
                        )
                    except Exception as e:
                        logger.debug(f"Could not plot MNI template for slice {z_idx}: {e}")

                # Apply threshold (set values below threshold to NaN for transparency)
                # Only mask if threshold > 0, otherwise show all data (for unthresholded maps)
                slice_data_masked = slice_data.copy()
                if threshold > 0:
                    slice_data_masked[np.abs(slice_data_masked) < threshold] = np.nan

                # Plot statistical map overlay with colormap matching nilearn
                im = ax.imshow(
                    slice_data_masked.T,
                    cmap=nilearn_cm.cold_hot,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    interpolation="bilinear",
                )
                ax.set_title(f"Z = {z_idx}", fontsize=10, fontweight="bold")
                ax.axis("off")

            # Remove unused subplots
            for idx in range(len(z_indices), len(axes)):
                fig.delaxes(axes[idx])

            # Add colorbar
            if im is not None:
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label("t-value", rotation=270, labelpad=15)

            # Add title
            if title:
                fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='This figure includes Axes that are not compatible with tight_layout'
                )
                fig.tight_layout(rect=[0, 0, 0.9, 0.99])

        except Exception as e:
            logger.warning(f"Could not plot axial slices: {e}")
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Error plotting axial slices: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

        fig_key = self._fig_to_base64(fig)
        plt.close(fig)

        return fig_key

    def _plot_stat_map(
        self,
        stat_map: nib.Nifti1Image,
        title: str = "",
        threshold: Optional[float] = None,
        colorbar: bool = True,
        cmap: str = "cold_hot",
    ) -> str:
        """Plot statistical map and return base64-encoded figure."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Three views
        display_modes = ["x", "y", "z"]
        cut_coords = [0, 0, 0]

        for ax, display_mode, cut in zip(axes, display_modes, cut_coords):
            try:
                # Suppress nilearn warnings about non-finite values
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Non-finite values detected')
                    plotting.plot_stat_map(
                        stat_map,
                        display_mode=display_mode,
                        cut_coords=[cut],
                        threshold=threshold,
                        colorbar=colorbar and (display_mode == "z"),
                        cmap=cmap,
                        axes=ax,
                        title=title if display_mode == "y" else "",
                    )
            except Exception as e:
                logger.warning(f"Could not plot stat map view {display_mode}: {e}")
                ax.text(0.5, 0.5, f"Error plotting {display_mode} view",
                       ha='center', va='center', transform=ax.transAxes)

        # Suppress tight_layout warnings from matplotlib
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
            plt.tight_layout()

        fig_key = self._fig_to_base64(fig)
        plt.close(fig)

        return fig_key
    
    def _compute_color_scale(self, stat_map: nib.Nifti1Image, threshold: float = 1.96) -> tuple:
        """
        Compute consistent vmin/vmax for color scaling from statistical map.

        Uses nilearn-like logic: computes bounds from values above the threshold,
        with symmetric scaling for bidirectional colormaps.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map to analyze.
        threshold : float
            Threshold for identifying displayed values. Only values with
            absolute value >= threshold are used for computing bounds.

        Returns
        -------
        tuple
            (vmin, vmax) for consistent coloring with proper symmetry.
        """
        data = stat_map.get_fdata()

        # Get valid (non-NaN) data
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            return -1.0, 1.0

        # Apply threshold: only consider values above threshold for bounds
        # This matches nilearn's approach of excluding sub-threshold values
        thresholded_data = valid_data[np.abs(valid_data) >= threshold]

        # If all values are below threshold, use full data for bounds
        if len(thresholded_data) == 0:
            thresholded_data = valid_data

        # Compute bounds from displayed data (robust percentile approach)
        # Use 5-95 percentile for positive and negative values
        abs_data = np.abs(thresholded_data)
        percentile_val = np.percentile(abs_data, 99)

        # For bidirectional colormaps (cold_hot, RdBu), use symmetric scaling
        # Find the max of positive and negative extremes
        pos_max = np.percentile(thresholded_data[thresholded_data > 0], 99) if np.any(thresholded_data > 0) else percentile_val
        neg_min = np.percentile(thresholded_data[thresholded_data < 0], 1) if np.any(thresholded_data < 0) else -percentile_val

        # Slightly increase the extremes for visual purposes only
        pos_max = 1.2*pos_max
        neg_min = 1.2*neg_min

        # Debug logging
        logger.debug(f"Color scale computation:")
        logger.debug(f"  Threshold used: {threshold}")
        logger.debug(f"  Valid data points: {len(valid_data)}")
        logger.debug(f"  Thresholded data points: {len(thresholded_data)}")
        logger.debug(f"  Positive values: {np.sum(thresholded_data > 0)}")
        logger.debug(f"  Negative values: {np.sum(thresholded_data < 0)}")
        logger.debug(f"  95th percentile of positive: {pos_max:.4f}")
        logger.debug(f"  5th percentile of negative: {neg_min:.4f}")

        # Make symmetric for better visualization with bidirectional colormap
        abs_max = max(abs(pos_max), abs(neg_min))

        logger.debug(f"  Final symmetric scale: ±{abs_max:.4f}")

        # Ensure minimum range to avoid grey appearance
        if abs_max < 0.1:
            logger.debug(f"  Scale too small ({abs_max:.4f}), setting to ±1.0")
            abs_max = 1.0

        logger.info(f"Computed color scale: vmin={-abs_max:.4f}, vmax={abs_max:.4f}")
        return -abs_max, abs_max

    def _plot_glass_brain(
        self,
        stat_map: nib.Nifti1Image,
        title: str = "",
        threshold: float = 3.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple:
        """
        Plot glass brain and return base64-encoded figure and color scale.

        Parameters
        ----------
        stat_map : nibabel.Nifti1Image
            Statistical map to plot.
        title : str
            Title for the plot.
        threshold : float
            Threshold for visualization.
        vmin : float, optional
            Minimum value for color scale. If None, computed from data.
        vmax : float, optional
            Maximum value for color scale. If None, computed from data.

        Returns
        -------
        tuple
            (fig_key, vmin, vmax) where fig_key is base64 string and vmin/vmax are the color scales.
        """
        # Compute color scale if not provided
        if vmin is None or vmax is None:
            vmin, vmax = self._compute_color_scale(stat_map, threshold=threshold)

        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            plotting.plot_glass_brain(
                stat_map,
                threshold=threshold,
                colorbar=True,
                plot_abs=False,
                title=title,
                axes=ax,
                vmin=vmin,
                vmax=vmax,
                cmap='cold_hot',
            )
        except Exception as e:
            logger.warning(f"Could not plot glass brain: {e}")
            ax.text(0.5, 0.5, "Error plotting glass brain",
                   ha='center', va='center', transform=ax.transAxes)

        # Suppress tight_layout warnings from matplotlib
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
            plt.tight_layout()

        fig_key = self._fig_to_base64(fig)
        plt.close(fig)

        return fig_key, vmin, vmax

    def _plot_connectivity_matrix(
        self,
        matrix: np.ndarray,
        title: str = "",
        roi_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "RdBu_r",
    ) -> str:
        """
        Plot connectivity matrix as a heatmap.

        Parameters
        ----------
        matrix : np.ndarray
            2D connectivity or statistical matrix (n_rois x n_rois).
        title : str
            Title for the plot.
        roi_names : list of str, optional
            Names of ROIs for axis labels.
        threshold : float, optional
            Threshold to mask values. Values with |x| < threshold will be shown as 0.
        vmin : float, optional
            Minimum value for color scale.
        vmax : float, optional
            Maximum value for color scale.
        cmap : str
            Colormap name.

        Returns
        -------
        str
            Base64-encoded figure key.
        """
        n_rois = matrix.shape[0]
        
        # Apply threshold if provided
        if threshold is not None:
            plot_matrix = matrix.copy()
            plot_matrix[np.abs(plot_matrix) < threshold] = 0
        else:
            plot_matrix = matrix
        
        # Determine color scale
        if vmin is None and vmax is None:
            vmax = np.nanmax(np.abs(plot_matrix))
            vmin = -vmax
        
        # Figure size scales with number of ROIs
        fig_size = max(8, min(20, n_rois / 5))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Create heatmap
        im = ax.imshow(
            plot_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('t-statistic', fontsize=10)
        
        # Set axis labels
        if roi_names is not None and n_rois <= 50:
            ax.set_xticks(np.arange(n_rois))
            ax.set_yticks(np.arange(n_rois))
            ax.set_xticklabels(roi_names, rotation=90, fontsize=6)
            ax.set_yticklabels(roi_names, fontsize=6)
        else:
            ax.set_xlabel('ROI index', fontsize=10)
            ax.set_ylabel('ROI index', fontsize=10)
        
        ax.set_title(title, fontsize=12)
        
        plt.tight_layout()
        fig_key = self._fig_to_base64(fig)
        plt.close(fig)
        
        return fig_key

    def _plot_connectome(
        self,
        matrix: np.ndarray,
        coordinates: np.ndarray,
        title: str = "",
        threshold: Optional[float] = None,
        edge_vmin: Optional[float] = None,
        edge_vmax: Optional[float] = None,
        node_size: float = 50.0,
        node_color: str = "auto",
    ) -> str:
        """
        Plot connectome on glass brain.

        Parameters
        ----------
        matrix : np.ndarray
            Connectivity or statistical matrix (n_rois x n_rois).
        coordinates : np.ndarray
            ROI coordinates in MNI space (n_rois x 3).
        title : str
            Title for the plot.
        threshold : float, optional
            Threshold for edge visibility.
        edge_vmin : float, optional
            Minimum value for edge color scale.
        edge_vmax : float, optional
            Maximum value for edge color scale.
        node_size : float
            Size of nodes.
        node_color : str
            Color for nodes. "auto" uses degree coloring.

        Returns
        -------
        str
            Base64-encoded figure key.
        """
        # Apply threshold if provided
        if threshold is not None and np.isfinite(threshold):
            plot_matrix = matrix.copy()
            plot_matrix[np.abs(plot_matrix) < threshold] = 0
        elif threshold is not None and not np.isfinite(threshold):
            # Infinite threshold means nothing is significant - show empty matrix
            plot_matrix = np.zeros_like(matrix)
        else:
            plot_matrix = matrix
        
        # Count non-zero edges for debugging
        n_nonzero = np.count_nonzero(plot_matrix)
        logger.debug(f"Connectome plot: {n_nonzero} non-zero edges in matrix")
        
        # Determine edge color scale
        if edge_vmin is None and edge_vmax is None:
            nonzero_vals = plot_matrix[np.isfinite(plot_matrix) & (plot_matrix != 0)]
            if len(nonzero_vals) > 0:
                edge_vmax = np.max(np.abs(nonzero_vals))
                edge_vmin = -edge_vmax
            else:
                edge_vmin, edge_vmax = -1, 1
        
        logger.debug(f"Connectome plot: edge_vmin={edge_vmin}, edge_vmax={edge_vmax}, threshold={threshold}")
        
        fig = plt.figure(figsize=(14, 5))
        
        # Three views: sagittal, coronal, axial
        views = ['x', 'y', 'z']
        view_titles = ['Sagittal', 'Coronal', 'Axial']
        
        for i, (view, view_title) in enumerate(zip(views, view_titles)):
            ax = fig.add_subplot(1, 3, i + 1)
            try:
                # Avoid passing non-finite threshold to nilearn
                plot_threshold = threshold if (threshold is not None and np.isfinite(threshold)) else None
                plotting.plot_connectome(
                    plot_matrix,
                    coordinates,
                    node_size=node_size,
                    edge_threshold=plot_threshold,
                    edge_vmin=edge_vmin,
                    edge_vmax=edge_vmax,
                    edge_cmap='cold_hot',
                    display_mode=view,
                    axes=ax,
                    title=view_title if i == 0 else None,
                    colorbar=(i == 2),  # Only show colorbar on last plot
                )
            except Exception as e:
                logger.warning(f"Could not plot connectome view {view}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:30]}",
                       ha='center', va='center', transform=ax.transAxes)
        
        fig.suptitle(title, fontsize=12, y=1.02)
        plt.tight_layout()
        
        fig_key = self._fig_to_base64(fig)
        plt.close(fig)
        
        return fig_key

    def _plot_edge_histogram(
        self,
        t_edges: np.ndarray,
        p_edges: np.ndarray,
        threshold: Optional[float] = None,
        alpha: float = 0.05,
        title: str = "Edge t-statistic Distribution",
    ) -> str:
        """
        Plot histogram of edge t-statistics.

        Parameters
        ----------
        t_edges : np.ndarray
            1D array of t-statistics for all edges.
        p_edges : np.ndarray
            1D array of p-values for all edges.
        threshold : float, optional
            Threshold to show as vertical lines.
        alpha : float
            Significance level for annotation.
        title : str
            Title for the plot.

        Returns
        -------
        str
            Base64-encoded figure key.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Filter out NaN/Inf values before plotting
        t_valid = t_edges[np.isfinite(t_edges)]
        p_valid = p_edges[np.isfinite(p_edges)]
        
        n_nan_t = len(t_edges) - len(t_valid)
        n_nan_p = len(p_edges) - len(p_valid)
        if n_nan_t > 0:
            logger.warning(f"Excluded {n_nan_t} non-finite t-statistic values from histogram")
        if n_nan_p > 0:
            logger.warning(f"Excluded {n_nan_p} non-finite p-values from histogram")
        
        # t-statistic histogram
        if len(t_valid) > 0:
            ax1.hist(t_valid, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, 'No finite t-statistics',
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_xlabel('t-statistic')
        ax1.set_ylabel('Number of edges')
        ax1.set_title('t-statistic Distribution')
        
        if threshold is not None and np.isfinite(threshold):
            ax1.axvline(threshold, color='red', linestyle='--', label=f'Threshold: ±{threshold:.2f}')
            ax1.axvline(-threshold, color='red', linestyle='--')
            ax1.legend()
        
        # Count significant edges (using original arrays, NaN comparisons return False)
        n_sig_pos = np.sum(t_edges > (threshold if threshold else 0))
        n_sig_neg = np.sum(t_edges < -(threshold if threshold else 0))
        
        annotation = f'Sig+: {n_sig_pos}\nSig-: {n_sig_neg}'
        if n_nan_t > 0:
            annotation += f'\nNaN: {n_nan_t}'
        ax1.text(0.95, 0.95, annotation,
                transform=ax1.transAxes, va='top', ha='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # p-value histogram
        if len(p_valid) > 0:
            ax2.hist(p_valid, bins=50, color='coral', edgecolor='white', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'No finite p-values',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlabel('p-value')
        ax2.set_ylabel('Number of edges')
        ax2.set_title('p-value Distribution')
        ax2.axvline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
        ax2.legend()
        
        # Add annotation for significant edges
        n_sig = np.sum(p_edges < alpha)
        annotation_p = f'p < {alpha}: {n_sig} edges'
        if n_nan_p > 0:
            annotation_p += f'\nNaN: {n_nan_p}'
        ax2.text(0.95, 0.95, annotation_p,
                transform=ax2.transAxes, va='top', ha='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        
        fig_key = self._fig_to_base64(fig)
        plt.close(fig)
        
        return fig_key

    def add_connectivity_results_section(
        self,
        t_matrix: np.ndarray,
        p_matrix: np.ndarray,
        contrast_name: str,
        correction: str,
        threshold: float,
        coordinates: Optional[np.ndarray] = None,
        roi_names: Optional[List[str]] = None,
        atlas_name: Optional[str] = None,
        edge_table: Optional[pd.DataFrame] = None,
        df: Optional[int] = None,
        t_matrix_unthresholded: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a section for connectivity analysis results.

        Parameters
        ----------
        t_matrix : np.ndarray
            T-statistic matrix (may be thresholded or unthresholded depending on usage).
        p_matrix : np.ndarray
            P-value matrix.
        contrast_name : str
            Name of the contrast.
        correction : str
            Correction method (e.g., "fdr", "bonferroni", "uncorrected").
        threshold : float
            Threshold used for significance.
        coordinates : np.ndarray, optional
            ROI coordinates for connectome plot.
        roi_names : list of str, optional
            Names of ROIs.
        atlas_name : str, optional
            Name of the atlas used.
        edge_table : pd.DataFrame, optional
            Table of significant edges.
        df : int, optional
            Degrees of freedom.
        t_matrix_unthresholded : np.ndarray, optional
            Unthresholded t-statistic matrix. If provided and this is the first
            correction method, will display both unthresholded and thresholded matrices.
        """
        section_title = f"Connectivity Results: {contrast_name} ({correction.upper()})"
        
        # Compute color scale for consistent visualization
        finite_vals = np.abs(t_matrix[np.isfinite(t_matrix)])
        if len(finite_vals) > 0:
            vmax = np.max(finite_vals)
        else:
            vmax = 1.0
        vmin = -vmax
        
        # Start building section content
        content = f"""
        <p><strong>Contrast:</strong> {contrast_name}</p>
        <p><strong>Correction:</strong> {correction.upper()}</p>
        <p><strong>Threshold (t):</strong> {threshold:.4f}</p>
        """
        
        if df is not None:
            content += f"<p><strong>Degrees of freedom:</strong> {df}</p>\n"
        
        if atlas_name is not None:
            content += f"<p><strong>Atlas:</strong> {atlas_name}</p>\n"
        
        # Count significant edges
        triu_indices = np.triu_indices(t_matrix.shape[0], k=1)
        sig_mask = np.abs(t_matrix[triu_indices]) >= threshold
        n_sig = np.sum(sig_mask)
        n_total = len(triu_indices[0])
        n_pos = np.sum(t_matrix[triu_indices][sig_mask] > 0)
        n_neg = np.sum(t_matrix[triu_indices][sig_mask] < 0)
        
        content += f"""
        <p><strong>Significant edges:</strong> {n_sig} / {n_total} ({100*n_sig/n_total:.1f}%)</p>
        <p>Positive: {n_pos}, Negative: {n_neg}</p>
        """
        
        # Add unthresholded matrix heatmap (only once, for the first correction)
        if t_matrix_unthresholded is not None:
            content += "<h3>Unthresholded t-statistic Matrix</h3>\n"
            fig_key = self._plot_connectivity_matrix(
                t_matrix_unthresholded,
                title=f"{contrast_name} - Unthresholded",
                roi_names=roi_names,
                threshold=None,
                vmin=vmin,
                vmax=vmax,
            )
            content += f'<div class="figure"><img src="data:image/png;base64,{self._figures[fig_key]}" alt="Unthresholded {section_title}"></div>\n'
        
        # Add thresholded matrix heatmap
        content += "<h3>Thresholded t-statistic Matrix</h3>\n"
        fig_key = self._plot_connectivity_matrix(
            t_matrix,
            title=f"{contrast_name} - {correction.upper()} (t ≥ {threshold:.2f})",
            roi_names=roi_names,
            threshold=None,  # Matrix is already thresholded, don't apply threshold again
            vmin=vmin,
            vmax=vmax,
        )
        content += f'<div class="figure"><img src="data:image/png;base64,{self._figures[fig_key]}" alt="{section_title}"></div>\n'
        
        # Add connectome plot if coordinates available
        if coordinates is not None:
            content += "<h3>Connectome (Glass Brain)</h3>\n"
            try:
                fig_key = self._plot_connectome(
                    t_matrix,
                    coordinates,
                    title=f"{contrast_name} - {correction.upper()}",
                    threshold=None,  # Matrix is already thresholded, don't apply threshold again
                    edge_vmin=vmin,
                    edge_vmax=vmax,
                )
                content += f'<div class="figure"><img src="data:image/png;base64,{self._figures[fig_key]}" alt="Connectome"></div>\n'
            except Exception as e:
                logger.warning(f"Could not generate connectome plot: {e}")
                content += f"<p><em>Connectome plot unavailable: {e}</em></p>\n"
        
        # Add edge histogram
        t_edges = t_matrix[triu_indices]
        p_edges = p_matrix[triu_indices]
        content += "<h3>Edge Statistics Distribution</h3>\n"
        fig_key = self._plot_edge_histogram(
            t_edges,
            p_edges,
            threshold=threshold,
            title=f"{contrast_name} - Edge Distribution",
        )
        content += f'<div class="figure"><img src="data:image/png;base64,{self._figures[fig_key]}" alt="Edge histogram"></div>\n'
        
        # Add edge table if available
        if edge_table is not None and len(edge_table) > 0:
            content += "<h3>Significant Edges</h3>\n"
            # Show top 50 edges
            display_table = edge_table.head(50)
            content += display_table.to_html(index=False, classes="dataframe")
            if len(edge_table) > 50:
                content += f"<p><em>Showing top 50 of {len(edge_table)} significant edges</em></p>\n"
        
        self.add_section(section_title, content, section_type="html", level=2)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        
        # Store and return key
        fig_key = f"fig_{len(self._figures)}"
        self._figures[fig_key] = img_base64
        
        return fig_key
    
    def generate_html(self) -> str:
        """
        Generate the HTML report.

        Returns
        -------
        str
            HTML content.
        """
        # Build nested sections tree for reliable hierarchical rendering
        nested_sections = self._nest_sections(self.sections)

        # Use Jinja2 template if available
        if self.env is not None:
            try:
                template = self.env.get_template("report.html")
                html = template.render(
                    title=self.title,
                    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    sections=self.sections,
                    nested_sections=nested_sections,
                    figures=self._figures,
                    latex_tables=self._latex_tables,
                    tsv_data=self._tsv_data,
                )
                # Strip any leading/trailing whitespace to remove stray content
                return html.strip()
            except Exception as e:
                logger.warning(f"Could not use Jinja2 template: {e}")

        # Fallback: generate basic HTML
        return self._generate_basic_html()

    def _nest_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert flat list of sections with 'level' into a nested tree structure.

        Each node will be a dict with keys: title, content, type, level, caption, children (list).
        """
        root: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = []

        for sec in sections:
            node = {
                "title": sec.get("title", ""),
                "content": sec.get("content", ""),
                "type": sec.get("type", "text"),
                "level": int(sec.get("level", 1)),
                "caption": sec.get("caption", ""),
                "children": [],
            }

            # If stack empty, append to root
            if not stack:
                root.append(node)
                stack.append(node)
                continue

            # Pop stack until we find a parent with level < node.level
            while stack and stack[-1]["level"] >= node["level"]:
                stack.pop()

            if not stack:
                # No parent found, append to root
                root.append(node)
                stack.append(node)
            else:
                # Found parent
                stack[-1]["children"].append(node)
                stack.append(node)

        return root
    
    def _generate_basic_html(self) -> str:
        """Generate basic HTML without Jinja2 templates (modern theme fallback)."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background-color: #ffffff;
            color: #333;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}

        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}

        .metadata {{
            color: rgba(255, 255, 255, 0.95);
            font-size: 1rem;
            opacity: 0.9;
        }}

        .section {{
            background: #f9f9f9;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}

        .section:hover {{
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }}

        .section h2 {{
            color: #2d3748;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #667eea;
        }}

        .section h3 {{
            color: #2d3748;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}

        .summary-card {{
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}

        .summary-card:hover {{
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }}

        .summary-card.success {{
            border-left-color: #28a745;
            background: #d4edda;
        }}

        .summary-card.warning {{
            border-left-color: #ffc107;
            background: #fff3cd;
        }}

        .summary-card.info {{
            border-left-color: #667eea;
            background: #d1ecf1;
        }}

        .summary-card h4 {{
            color: #2d3748;
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}

        .summary-card .value {{
            font-size: 2.2rem;
            font-weight: bold;
            color: #667eea;
        }}

        .figure {{
            text-align: center;
            margin: 2rem 0;
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 8px;
        }}

        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
            transition: transform 0.3s ease;
        }}

        .figure img:hover {{
            transform: scale(1.02);
        }}

        .figure-caption {{
            margin-top: 1rem;
            padding: 0.75rem;
            background-color: #f9f9f9;
            border-left: 3px solid #667eea;
            color: #495057;
            font-size: 0.9rem;
            line-height: 1.5;
            border-radius: 3px;
        }}

        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            overflow: hidden;
        }}

        .table thead {{
            background: #667eea;
            color: white;
        }}

        .table th {{
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}

        .table td {{
            padding: 0.9rem 1rem;
            border-bottom: 1px solid #e2e8f0;
        }}

        .table tbody tr:hover {{
            background-color: #f7fafc;
        }}

        .table tbody tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .info-box {{
            background: #d1ecf1;
            border-left: 4px solid #667eea;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 4px;
            color: #0c5460;
        }}

        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 4px;
        }}

        .success-box {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: 4px;
        }}

        ul {{
            padding-left: 1.5rem;
            margin: 1rem 0;
        }}

        li {{
            margin-bottom: 0.5rem;
            line-height: 1.8;
        }}

        code {{
            background: #f9f9f9;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #667eea;
            border: 1px solid #e2e8f0;
        }}

        .file-list {{
            max-height: 400px;
            overflow-y: auto;
            background: #f9f9f9;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }}

        .file-list-item {{
            padding: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
        }}

        .file-list-item:last-child {{
            border-bottom: none;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            background: #2d3748;
            color: #ffffff;
            font-size: 0.9rem;
            margin-top: 3rem;
            border-radius: 8px;
        }}

        footer p {{
            margin: 5px 0;
        }}

        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8em;
            }}

            .summary-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <p class="metadata">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
"""
        
        for section in self.sections:
            html += f"""
        <div class="section">
            <h2>{section['title']}</h2>
"""
            
            if section["type"] == "figure":
                fig_key = section["content"]
                if fig_key in self._figures:
                    html += f"""
            <div class="figure">
                <img src="data:image/png;base64,{self._figures[fig_key]}" alt="{section['title']}">
            </div>
"""
            elif section["type"] == "table":
                html += section["content"]
            else:
                html += section["content"]
            
            html += """
        </div>
"""

        html += """
        <footer>
            <p>&copy; StatCraft - BIDS-based Second-Level Neuroimaging Analysis Tool</p>
            <p><a href="https://github.com/ln2t/StatCraft" style="color: #667eea; text-decoration: none;">Documentation & Source Code</a></p>
        </footer>
    </div>
</body>
</html>
"""

        return html
    
    def save(
        self,
        filename: str = "report.html",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save the HTML report to file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        output_dir : str or Path, optional
            Output directory. Uses self.output_dir if not provided.
        
        Returns
        -------
        Path
            Path to saved report.
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        if output_dir is None:
            output_dir = Path.cwd()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        html = self.generate_html()
        filepath.write_text(html)
        
        logger.info(f"Report saved to: {filepath}")
        return filepath
