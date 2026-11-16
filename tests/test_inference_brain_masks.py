import json

import nibabel as nib
import numpy as np
import pytest

from statcraft.pipeline import StatCraftPipeline


def save(path, data, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32),
                             np.eye(4) if affine is None else affine), path)


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "inputs"
    data.mkdir()
    (data / "dataset_description.json").write_text(json.dumps({
        "Name": "Mask test", "BIDSVersion": "1.8.0",
    }))
    (data / "participants.tsv").write_text(
        "participant_id\n" + "".join(f"sub-{subject:02}\n" for subject in range(1, 5)))
    for subject in range(1, 5):
        for condition in ("A", "B"):
            values = np.random.default_rng(subject * 3 + (condition == "B")).normal(
                size=(4, 4, 4)) + subject
            save(data / f"sub-{subject:02}_desc-{condition}_stat.nii.gz", values)
        mask = np.ones((4, 4, 4))
        mask[subject - 1, 0, 0] = 0
        save(data / f"sub-{subject:02}_desc-brain_mask.nii.gz", mask)
    return data


@pytest.mark.parametrize("route", ["one-sample", "two-sample", "paired", "combine", "legacy"])
@pytest.mark.parametrize("fallback", [False, True])
def test_all_routes_use_intersection_for_maps_and_inference(
        inputs, tmp_path, route, fallback, monkeypatch, caplog):
    if fallback:
        import nilearn.datasets

        (inputs / "sub-04_desc-brain_mask.nii.gz").unlink()
        template = np.ones((4, 4, 4), dtype=np.uint8)
        template[3, 3, 3] = 0
        monkeypatch.setattr(nilearn.datasets, "load_mni152_brain_mask",
                            lambda threshold: nib.Nifti1Image(template, np.eye(4)))
    patterns = {"A": "*desc-A_stat.nii.gz", "B": "*desc-B_stat.nii.gz"}
    paired = {"sample_patterns": patterns, "pair_by": "sub"}
    if route == "combine":
        paired["combine"] = "A-B"
    if route == "legacy":
        # A/B are also available as session entities for the condition-based route.
        for path in list(inputs.glob("*_stat.nii.gz")):
            path.rename(path.with_name(path.name.replace("_desc-", "_ses-")))
        paired = {"pair_by": "session", "condition1": "A", "condition2": "B"}
    analysis = "paired" if route in ("paired", "combine", "legacy") else route
    config = {
        "analysis_type": analysis,
        "paired_test": paired,
        "output": {"generate_report": False},
        "inference": {"corrections": ["uncorrected"], "cluster_threshold": 0},
        "verbose": 0,
    }
    if route == "two-sample":
        config["sample_patterns"] = patterns
    pipeline = StatCraftPipeline(inputs, tmp_path / "out", [], config=config)
    pipeline.annotate_clusters = lambda: {}
    if route in ("two-sample", "paired", "combine"):
        result = pipeline.run(sample_patterns=patterns)
    else:
        result = pipeline.run(pattern="*_stat.nii.gz" if route == "legacy" else patterns["A"])

    domain = pipeline.glm.mask.get_fdata().astype(bool)
    expected = np.ones((4, 4, 4), dtype=bool)
    if fallback:
        expected[3, 3, 3] = False
        assert "using the MNI152 brain mask instead of individual masks" in caplog.text
    else:
        expected[:, 0, 0] = False
    np.testing.assert_array_equal(domain, expected)
    np.testing.assert_array_equal(nib.load(result["saved_files"]["inference_mask"]).get_fdata(), domain)
    assert pipeline.inference.mask is pipeline.glm.mask
    for contrast in pipeline.glm.results.values():
        for image in contrast["maps"].values():
            assert np.all(image.get_fdata()[~domain] == 0)
    for key, path in result["saved_files"].items():
        if str(path).endswith(".nii.gz") and key != "inference_mask":
            assert np.all(nib.load(path).get_fdata()[~domain] == 0)
    for maps in pipeline.inference.thresholded_maps.values():
        for image in maps.values():
            assert np.all(image.get_fdata()[~domain] == 0)


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_one_mask_uses_only_mni_and_warns(inputs, tmp_path, monkeypatch, caplog, explicit):
    import nilearn.datasets

    (inputs / "sub-04_desc-brain_mask.nii.gz").unlink()
    mni = np.ones((4, 4, 4), dtype=np.uint8)
    mni[3, 3, 3] = 0
    monkeypatch.setattr(nilearn.datasets, "load_mni152_brain_mask",
                        lambda threshold: nib.Nifti1Image(mni, np.eye(4)))
    config = {"analysis_type": "one-sample", "verbose": 0}
    if explicit:
        config["inference_mask"] = str(inputs / "sub-*_desc-brain_mask.nii.gz")
    pipeline = StatCraftPipeline(inputs, tmp_path / "out", [], config=config)
    infos = pipeline.data_loader.get_images(pattern="*desc-A_stat.nii.gz")
    result = pipeline._build_inference_mask(infos)
    np.testing.assert_array_equal(result.get_fdata(), mni)
    assert "Missing inference brain mask for 1 of 4" in caplog.text
    assert "using the MNI152 brain mask instead of individual masks" in caplog.text
    assert pipeline._inference_mask_metadata["MaskSources"] == []
    assert len(pipeline._inference_mask_metadata["MissingMaskInputSources"]) == 1


def test_mni_fallback_resamples_to_input_grid(tmp_path):
    from nilearn.datasets import load_mni152_brain_mask
    from nilearn.image import resample_to_img

    reference = tmp_path / "sub-01_stat.nii.gz"
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = [65, -10, -10]
    save(reference, np.ones((10, 10, 10)), affine)
    pipeline = StatCraftPipeline(tmp_path, tmp_path / "out", [],
                                 config={"analysis_type": "one-sample", "verbose": 0})
    result = pipeline._build_inference_mask([{"path": reference, "subject": "01",
                                               "entities": {"subject": "01"}}])
    expected = resample_to_img(load_mni152_brain_mask(), nib.load(reference),
                               interpolation="nearest").get_fdata() > 0
    np.testing.assert_array_equal(result.get_fdata(), expected)
    np.testing.assert_allclose(result.affine, affine)
    assert expected.any() and not expected.all()


def test_ambiguous_individual_masks_are_not_silently_selected(inputs, tmp_path):
    save(inputs / "sub-01_acq-extra_desc-brain_mask.nii.gz", np.ones((4, 4, 4)))
    # Two masks sharing the input's entities must not be arbitrarily chosen.
    save(inputs / "sub-01_desc-brain_mask.nii", np.ones((4, 4, 4)))
    pipeline = StatCraftPipeline(inputs, tmp_path / "out", [],
                                 config={"analysis_type": "one-sample", "verbose": 0})
    infos = pipeline.data_loader.get_images(pattern="*desc-A_stat.nii.gz")
    with pytest.raises(ValueError, match="Ambiguous mask"):
        pipeline._build_inference_mask(infos)


def test_disconnected_components_and_zero_data_are_preserved(inputs, tmp_path):
    domain = np.zeros((4, 4, 4), dtype=np.uint8)
    domain[0, 0, 0] = domain[3, 3, 3] = 1
    for path in inputs.glob("*desc-brain_mask.nii.gz"):
        save(path, domain)
    first = next(inputs.glob("*desc-A_stat.nii.gz"))
    values = nib.load(first).get_fdata()
    values[0, 0, 0] = 0
    save(first, values)
    pipeline = StatCraftPipeline(inputs, tmp_path / "out", [],
                                 config={"analysis_type": "one-sample", "verbose": 0})
    infos = pipeline.data_loader.get_images(pattern="*desc-A_stat.nii.gz")
    np.testing.assert_array_equal(pipeline._build_inference_mask(infos).get_fdata(), domain)
