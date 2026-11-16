"""Regression tests exercise discovery -> pairing -> arithmetic -> GLM -> inference."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, ttest_1samp

from statcraft.pipeline import StatCraftPipeline


def save_image(path, values, translation=0):
    affine = np.eye(4)
    affine[0, 3] = translation
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=float), affine), path)
    return path


@pytest.fixture
def cohort(tmp_path):
    data = tmp_path / "inputs"
    data.mkdir()
    (data / "dataset_description.json").write_text(json.dumps({
        "Name": "Synthetic paired comparisons", "BIDSVersion": "1.8.0"}))
    for subject in range(1, 5):
        for offset, name in enumerate(("A", "B", "C")):
            values = np.random.default_rng(subject * 10 + offset).normal(size=(3, 3, 3))
            values += subject * 0.2 + offset
            # A-B and A+B-2*C are exactly zero here, and must remain in the mask.
            values[0, 0, 0] = 5
            save_image(data / f"sub-{subject:02}_ses-01_run-1_desc-{name}_cvr.nii.gz", values)
    mask = np.ones((3, 3, 3))
    mask[2, 2, 2] = 0
    mask_path = save_image(tmp_path / "domain.nii.gz", mask)
    return data, mask_path


def pipeline_for(cohort, tmp_path, combine=False, **overrides):
    data, mask_path = cohort
    patterns = {name: f"*desc-{name}_*" for name in (("A", "B", "C") if combine else ("A", "B"))}
    config = {
        "analysis_type": "paired",
        "paired_test": {"sample_patterns": patterns, "pair_by": "sub,ses,run",
                        "combine": "A+B-2*C" if combine else None},
        "inference_mask": str(mask_path),
        "output": {"generate_report": False},
        "inference": {"corrections": [], "cluster_threshold": 0},
        "verbose": 0,
    }
    config.update(overrides)
    pipeline = StatCraftPipeline(data, tmp_path / "out", [], config=config)
    # Annotation may download atlases; it does not alter any tested numeric path.
    pipeline.annotate_clusters = lambda: {}
    return pipeline, patterns


def run_route(pipeline, patterns, combine=False, **kwargs):
    return pipeline.run(sample_patterns=patterns, **kwargs)


def test_combine_task_sets_produce_distinct_persistent_outputs(cohort, tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import statcraft.pipeline as pipeline_module

    data, _ = cohort
    for path in list(data.glob("*desc-[AB]_cvr.nii.gz")):
        task = "task1" if "desc-A" in path.name else "task2"
        other = "task3" if task == "task1" else "task4"
        save_image(data / path.name.replace("_desc-", f"_task-{other}_desc-"),
                   nib.load(path).get_fdata())
        path.rename(data / path.name.replace("_desc-", f"_task-{task}_desc-"))

    def report_generator(*args, **kwargs):
        report = MagicMock()
        report._is_map_empty.return_value = True

        def save(filename):
            path = tmp_path / "out" / filename
            path.write_text("test report")
            return path

        report.save.side_effect = save
        return report

    monkeypatch.setattr(pipeline_module, "ReportGenerator", report_generator)
    saved_results = []
    prefixes = []
    for first, second in (("task1", "task2"), ("task3", "task4"), ("task1", "task4")):
        patterns = {"A": f"*task-{first}_*desc-A_*",
                    "B": f"*task-{second}_*desc-B_*"}
        pipeline, _ = pipeline_for(
            cohort, tmp_path, combine=True,
            paired_test={"sample_patterns": patterns, "pair_by": "sub,ses,run",
                         "combine": "A-B"},
            output={"generate_report": True, "save_supplementary_data": True},
        )
        result = pipeline.run(sample_patterns=patterns)
        name = pipeline._generate_bids_prefix()
        prefixes.append(name)
        saved_results.append(result["saved_files"])
        assert name.startswith(f"task-{first}{second}H")
        assert all(name in str(result["saved_files"][key])
                   for key in ("inference_mask", "A_minus_B_stat", "design_matrix", "config"))
        assert list((tmp_path / "out").glob(f"{name}_contrast-*_report.html"))
        diff_paths = [result["saved_files"][f"input_image_{i}"] for i in range(4)]
        assert len(set(diff_paths)) == 4
        for i, path in enumerate(diff_paths, 1):
            assert f"sub-{i:02}_" in path.name
            assert name.split("_")[0] in path.name
            np.testing.assert_array_equal(nib.load(path).get_fdata(),
                                          pipeline._diff_images[i - 1].get_fdata())

    assert len(set(prefixes)) == 3
    for saved in saved_results:
        assert all(path.exists() for path in saved.values())
    assert saved_results[0]["A_minus_B_stat"] != saved_results[1]["A_minus_B_stat"]
    assert saved_results[0]["design_matrix"] != saved_results[1]["design_matrix"]
    assert saved_results[0]["input_image_0"] != saved_results[2]["input_image_0"]


@pytest.mark.parametrize("route", ["paired", "legacy"])
def test_supplementary_difference_names_identify_correct_participants(cohort, tmp_path, route):
    data, mask_path = cohort
    for path in list(data.glob("*desc-[AB]_cvr.nii.gz")):
        task = "task1" if "desc-A" in path.name else "task2"
        if route == "legacy":
            condition = "pre" if "desc-A" in path.name else "post"
            name = path.name.replace("_ses-01_", f"_ses-{condition}_")
            name = name.replace("_desc-", f"_task-{task}_desc-")
        else:
            name = path.name.replace("_desc-", f"_task-{task}_desc-")
        path.rename(data / name)

    if route == "paired":
        pipeline, patterns = pipeline_for(cohort, tmp_path, output={
            "generate_report": False, "save_supplementary_data": True})
        result = pipeline.run(sample_patterns=patterns)
    else:
        pipeline = StatCraftPipeline(data, tmp_path / "out", [], config={
            "analysis_type": "paired",
            "paired_test": {"pair_by": "session", "condition1": "pre", "condition2": "post"},
            "inference_mask": str(mask_path),
            "output": {"generate_report": False, "save_supplementary_data": True},
            "inference": {"corrections": []},
            "verbose": 0,
        })
        pipeline.annotate_clusters = lambda: {}
        result = pipeline.run(pattern="*task-*_cvr.nii.gz")

    task = pipeline._generate_bids_prefix().split("_")[0]
    paths = [result["saved_files"][f"input_image_{i}"] for i in range(4)]
    assert len(set(paths)) == 4
    for subject, path in enumerate(paths, 1):
        assert f"sub-{subject:02}_" in path.name
        assert task in path.name
        np.testing.assert_array_equal(nib.load(path).get_fdata(),
                                      pipeline._diff_images[subject - 1].get_fdata())


@pytest.mark.parametrize("combine", [False, True])
def test_actual_routes_reject_duplicate_complete_keys(cohort, tmp_path, combine):
    data, _ = cohort
    original = next(data.glob("sub-01*desc-A*"))
    save_image(data / original.name.replace("_cvr", "_variant-extra_cvr"), np.ones((3, 3, 3)))
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine)
    with pytest.raises(ValueError, match="Duplicate pairing key"):
        run_route(pipeline, patterns, combine)


@pytest.mark.parametrize("combine", [False, True])
def test_actual_routes_exclude_whole_geometry_invalid_pair(cohort, tmp_path, combine):
    data, _ = cohort
    bad_path = next(data.glob("sub-04*desc-B*"))
    save_image(bad_path, np.full((3, 3, 3), 1e6), translation=20)
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine)
    result = run_route(pipeline, patterns, combine)
    assert len(pipeline._diff_images) == 3
    assert all(info["subject"] != "04" for info in result["images"])
    assert len(pipeline._invalid_images) == (3 if combine else 2)
    saved_exclusions = result["saved_files"]["excluded_files"].read_text()
    assert "Affine mismatch" in saved_exclusions
    assert "sub-04" in saved_exclusions


@pytest.mark.parametrize("combine", [False, True])
def test_actual_routes_finite_mask_keeps_zero_differences_and_matches_reference(cohort, tmp_path, combine):
    data, _ = cohort
    corrupted = next(data.glob("sub-03*desc-B*"))
    values = nib.load(corrupted).get_fdata()
    values[1, 1, 1] = np.nan
    save_image(corrupted, values)
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine)
    result = run_route(pipeline, patterns, combine)
    mask = pipeline.glm.mask.get_fdata().astype(bool)
    assert mask[0, 0, 0]  # zero differences are valid
    assert not mask[1, 1, 1]  # nonfinite in any original map
    assert not mask[2, 2, 2]  # predetermined anatomical exclusion
    assert int(mask.sum()) == 25
    assert np.array_equal(pipeline.glm.model.masker_.mask_img_.get_fdata(), mask)
    assert all(np.all(np.isfinite(img.get_fdata())) for img in pipeline._diff_images)
    expected = []
    for subject in range(1, 5):
        a = nib.load(next(data.glob(f"sub-{subject:02}*desc-A*"))).get_fdata()
        b = nib.load(next(data.glob(f"sub-{subject:02}*desc-B*"))).get_fdata()
        if combine:
            c = nib.load(next(data.glob(f"sub-{subject:02}*desc-C*"))).get_fdata()
            expected.append((a + b - 2*c)[mask])
        else:
            expected.append((a - b)[mask])
    expected = np.stack(expected)
    np.testing.assert_allclose(np.stack([img.get_fdata()[mask] for img in pipeline._diff_images]), expected)
    reference = ttest_1samp(expected, 0, axis=0).statistic
    contrast_name = next(iter(pipeline.glm.results))
    actual = pipeline.glm.results[contrast_name]["maps"]["stat"].get_fdata()[mask]
    np.testing.assert_allclose(actual[1:], reference[1:], rtol=1e-5)
    saved = result["saved_files"]
    np.testing.assert_array_equal(nib.load(saved["inference_mask"]).get_fdata(), mask)
    metadata = json.loads(saved["inference_mask_metadata"].read_text())
    assert metadata["VoxelCount"] == 25
    assert metadata["ZeroValuesAreValid"]
    assert not metadata["NormalizationMasksDefineInference"]


@pytest.mark.parametrize("combine", [False, True])
def test_actual_routes_reject_foreign_normalization_grid(cohort, tmp_path, combine):
    foreign = save_image(tmp_path / "foreign.nii.gz", np.ones((3, 3, 3)), translation=20)
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine)
    with pytest.raises((ValueError, RuntimeError), match="shape/affine mismatch"):
        run_route(pipeline, patterns, combine, scaling=str(foreign))


@pytest.mark.parametrize("combine", [False, True])
def test_normalization_roi_does_not_define_test_domain(cohort, tmp_path, combine):
    roi = np.zeros((3, 3, 3))
    roi[0, 0, 0] = 1
    roi_path = save_image(tmp_path / "normalization_roi.nii.gz", roi)
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine)
    result = run_route(pipeline, patterns, combine, scaling=str(roi_path))
    assert int(pipeline.glm.mask.get_fdata().sum()) == 26
    assert int(nib.load(result['saved_files']['normalization_mask']).get_fdata().sum()) == 1


def test_permutation_and_all_parametric_families_receive_identical_mask(cohort, tmp_path, monkeypatch):
    import statcraft.core.inference as inference_module

    threshold_masks, permutation_masks = [], []
    real_threshold = inference_module.threshold_stats_img
    real_permutation = inference_module.non_parametric_inference

    def threshold(*args, **kwargs):
        threshold_masks.append(kwargs.get("mask_img"))
        return real_threshold(*args, **kwargs)

    def permutation(*args, **kwargs):
        permutation_masks.append(kwargs.get("mask"))
        return real_permutation(*args, **kwargs)

    monkeypatch.setattr(inference_module, "threshold_stats_img", threshold)
    monkeypatch.setattr(inference_module, "non_parametric_inference", permutation)
    pipeline, patterns = pipeline_for(cohort, tmp_path, inference={
        "corrections": ["uncorrected", "fdr", "bonferroni"],
        "run_permutation": True, "n_permutations": 16, "cluster_threshold": 0,
    }, random_state=7)
    run_route(pipeline, patterns)
    assert len(threshold_masks) == 3
    assert len(permutation_masks) == 1
    assert all(mask is pipeline.glm.mask for mask in threshold_masks + permutation_masks)
    assert "A_minus_B_perm" in pipeline.inference._inference_results
    np.testing.assert_allclose(pipeline.inference.threshold_values["A_minus_B"]["uncorrected"],
                               norm.isf(0.001 / 2))


def test_missing_complete_entity_is_error(cohort, tmp_path):
    pipeline, patterns = pipeline_for(cohort, tmp_path,
                                     paired_test={"sample_patterns": {"A": "*desc-A*", "B": "*desc-B*"},
                                                  "pair_by": "sub,ses,echo"})
    with pytest.raises(ValueError, match="missing pairing entities.*echo"):
        run_route(pipeline, patterns)


def test_acquisition_and_resolution_aliases_agree_with_glob_parser(cohort, tmp_path):
    data, _ = cohort
    for path in data.glob('*.nii.gz'):
        path.rename(path.with_name(path.name.replace('_desc-', '_acq-test_res-2_desc-')))
    pipeline, patterns = pipeline_for(cohort, tmp_path,
                                     paired_test={'pair_by': 'sub,acq,res',
                                                  'sample_patterns': {'A': '*desc-A_*', 'B': '*desc-B_*'}})
    run_route(pipeline, patterns)
    assert len(pipeline._diff_images) == 4


def test_simple_ttest_rejects_repeated_subject_runs(cohort, tmp_path):
    data, _ = cohort
    for name in ("A", "B"):
        image = nib.load(next(data.glob(f"sub-01*desc-{name}*")))
        save_image(data / f"sub-01_ses-01_run-2_desc-{name}_cvr.nii.gz", image.get_fdata())
    pipeline, patterns = pipeline_for(cohort, tmp_path)
    with pytest.raises(ValueError, match="Repeated observations"):
        run_route(pipeline, patterns)


@pytest.mark.parametrize("combine", [False, True])
def test_actual_routes_reject_inference_mask_geometry(cohort, tmp_path, combine):
    foreign = save_image(tmp_path / "foreign_domain.nii.gz", np.ones((3, 3, 3)), translation=0.01)
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine, inference_mask=str(foreign))
    with pytest.raises(ValueError, match="Inference mask.*shape/affine mismatch"):
        run_route(pipeline, patterns, combine)


def test_actual_route_rejects_zscore_mask_geometry(cohort, tmp_path):
    foreign = save_image(tmp_path / "foreign_zscore.nii.gz", np.ones((3, 3, 3)), translation=10)
    pipeline, patterns = pipeline_for(cohort, tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="shape/affine mismatch"):
        run_route(pipeline, patterns, zscore=True, mask=str(foreign))


def test_participant_selection_is_applied_before_pairing(cohort, tmp_path):
    pipeline, patterns = pipeline_for(cohort, tmp_path)
    result = pipeline.run(sample_patterns=patterns, participant_label=["01", "02"])
    assert {info['subject'] for info in result['images']} == {"01", "02"}
    assert len(pipeline._diff_images) == 2


@pytest.mark.parametrize('normalize', [False, True])
def test_legacy_condition_route_uses_shared_mask_and_excludes_geometry_partner(cohort, tmp_path, normalize):
    data, mask = cohort
    for path in data.glob("*desc-C*"):
        path.unlink()
    bad_path = next(data.glob("sub-04*desc-B*"))
    save_image(bad_path, np.ones((3, 3, 3)), translation=20)
    pipeline = StatCraftPipeline(data, tmp_path / 'legacy_output', [], config={
        'analysis_type': 'paired', 'paired_test': {
            'pair_by': 'description', 'condition1': 'A', 'condition2': 'B'},
        'inference_mask': str(mask), 'output': {'generate_report': False},
        'inference': {'corrections': []}, 'verbose': 0,
    })
    pipeline.annotate_clusters = lambda: {}
    result = pipeline.run(pattern='*cvr.nii.gz', scaling=str(mask) if normalize else None)
    assert len(pipeline._diff_images) == 3
    assert all(info['subject'] != '04' for info in result['images'])
    assert int(pipeline.glm.mask.get_fdata().sum()) == 26
    assert len(pipeline._invalid_images) == 2
    if normalize:
        domain = nib.load(mask).get_fdata().astype(bool)
        a = nib.load(next(data.glob('sub-01*desc-A*'))).get_fdata()
        b = nib.load(next(data.glob('sub-01*desc-B*'))).get_fdata()
        np.testing.assert_allclose(pipeline._diff_images[0].get_fdata()[domain],
                                   (a/a[domain].mean()-b/b[domain].mean())[domain])


def test_cli_example_runs_full_pattern_path(cohort, tmp_path, monkeypatch):
    import sys
    from statcraft.cli import main

    data, mask = cohort
    monkeypatch.setattr(StatCraftPipeline, 'annotate_clusters', lambda self: {})
    monkeypatch.setattr(StatCraftPipeline, 'generate_report', lambda self: tmp_path / 'report.html')
    monkeypatch.setattr(sys, 'argv', [
        'statcraft', str(data), str(tmp_path / 'cli_output'), 'group',
        '--analysis-type', 'paired', '--patterns', 'A=*desc-A_* B=*desc-B_*',
        '--pair-by', 'sub,ses,run', '--inference-mask', str(mask),
    ])
    main()
    saved_mask = list((tmp_path / 'cli_output' / 'maps').glob('*desc-inference_mask.nii.gz'))
    assert len(saved_mask) == 1
    assert int(nib.load(saved_mask[0]).get_fdata().sum()) == 26


def test_invalid_paired_contrast_does_not_fall_back_to_default(cohort, tmp_path):
    pipeline, patterns = pipeline_for(cohort, tmp_path, contrast='Typo-A')
    with pytest.raises(ValueError, match='Paired contrast must be'):
        run_route(pipeline, patterns)


@pytest.mark.parametrize('combine', [False, True])
def test_normalization_uses_same_finite_voxels_for_all_methods(cohort, tmp_path, combine):
    data, _ = cohort
    path = next(data.glob('sub-01*desc-A*'))
    values = nib.load(path).get_fdata()
    values[1, 1, 1] = np.nan
    save_image(path, values)
    roi_path = save_image(tmp_path / 'normalization_all.nii.gz', np.ones((3, 3, 3)))
    pipeline, patterns = pipeline_for(cohort, tmp_path, combine)
    run_route(pipeline, patterns, combine, scaling=str(roi_path))
    support = pipeline._paired_normalization_support
    assert not support[1, 1, 1]
    assert support[2, 2, 2]  # normalization domain is independent of inference domain
    assert support.sum() == 26
    for subject, difference in zip(range(1, 5), pipeline._diff_images):
        normalized = []
        for name in ('A', 'B', 'C') if combine else ('A', 'B'):
            array = nib.load(next(data.glob(f'sub-{subject:02}*desc-{name}*'))).get_fdata()
            normalized.append(array / array[support].mean())
        expected = normalized[0] + normalized[1] - 2*normalized[2] if combine else normalized[0]-normalized[1]
        inference_support = pipeline.glm.mask.get_fdata().astype(bool)
        np.testing.assert_allclose(difference.get_fdata()[inference_support], expected[inference_support], rtol=1e-6)


def test_same_mask_for_normalization_and_inference_records_each_denominator(cohort, tmp_path):
    _, mask = cohort
    pipeline, patterns = pipeline_for(cohort, tmp_path, normalization={
        'mean_guard': {'minimum_absolute_mean_ratio': .01, 'require_positive_mean': False}})
    result = run_route(pipeline, patterns, scaling=str(mask))
    inference = nib.load(result['saved_files']['inference_mask']).get_fdata()
    normalization = nib.load(result['saved_files']['normalization_mask']).get_fdata()
    np.testing.assert_array_equal(inference, normalization)
    metadata = json.loads(result['saved_files']['inference_mask_metadata'].read_text())
    assert len(metadata['MeanNormalizationQC']) == 8
    for row in metadata['MeanNormalizationQC'].values():
        assert row['VoxelCount'] == 26
        assert abs(row['SignedMeanToMeanAbsoluteRatio']) >= .01
        assert row['MinimumAbsoluteMeanRatio'] == .01


def test_paired_route_stops_for_unstable_denominator_without_dropping_pair(cohort, tmp_path):
    data, mask = cohort
    values = np.zeros((3, 3, 3))
    inside = nib.load(mask).get_fdata().astype(bool)
    values[inside] = np.tile([1, -1], 13) + .0001
    save_image(next(data.glob('sub-01*desc-A*')), values)
    pipeline, patterns = pipeline_for(cohort, tmp_path, normalization={
        'mean_guard': {'minimum_absolute_mean_ratio': .01, 'require_positive_mean': True}})
    with pytest.raises(RuntimeError, match='Unstable normalization denominator'):
        run_route(pipeline, patterns, scaling=str(mask))


def test_config_only_cli_preserves_pairing_mask_inference_and_direction(cohort, tmp_path, monkeypatch):
    import sys
    import yaml
    from statcraft.cli import main

    data, mask = cohort
    config_path = tmp_path / 'comparison.yaml'
    config_path.write_text(yaml.safe_dump({
        'analysis_type': 'paired', 'inference_mask': str(mask), 'contrast': 'B-A',
        'paired_test': {'pair_by': ['subject', 'session', 'run'],
                        'sample_patterns': {'A': '*desc-A_*', 'B': '*desc-B_*'}},
        'inference': {'run_permutation': True, 'n_permutations': 16, 'alpha_uncorrected': 0.02,
                      'alpha_corrected': 0.1, 'cluster_threshold': 0},
        'glm': {'smoothing_fwhm': 0}, 'atlas': 'none', 'random_state': 11,
        'output': {'generate_report': False}, 'verbose': 0,
    }))
    out = tmp_path / 'config_output'
    monkeypatch.setattr(sys, 'argv', ['statcraft', str(data), str(out), 'group', '--config', str(config_path)])
    main()
    saved_configs = list(out.glob('logs/**/*config.yaml'))
    assert len(saved_configs) == 1
    saved = yaml.safe_load(saved_configs[0].read_text())
    assert saved['analysis_type'] == 'paired'
    assert saved['inference_mask'] == str(mask)
    assert saved['inference']['run_permutation']
    assert saved['inference']['n_permutations'] == 16
    assert saved['inference']['alpha_uncorrected'] == 0.02
    assert saved['inference']['alpha_corrected'] == 0.1
    assert saved['contrast'] == 'B-A'
    assert not saved['output']['generate_report']
    assert any(out.glob('maps/*B_minus_A*stat-fwer_perm*'))
    effect = next(out.glob('maps/*B_minus_A*stat-effect_size.nii.gz'))
    expected = np.mean([nib.load(next(data.glob(f'sub-{subject:02}*desc-B*'))).get_fdata()
                        - nib.load(next(data.glob(f'sub-{subject:02}*desc-A*'))).get_fdata()
                        for subject in range(1, 5)], axis=0)
    domain = nib.load(mask).get_fdata().astype(bool)
    np.testing.assert_allclose(nib.load(effect).get_fdata()[domain], expected[domain])
