import pytest

from statcraft.core.data_loader import DataLoader


def image_info(condition, path):
    return {
        "subject": "01",
        "path": str(path),
        "entities": {"session": condition},
    }


def test_ordinary_pairing_rejects_duplicate_condition(tmp_path):
    loader = object.__new__(DataLoader)
    images = [
        image_info("pre", tmp_path / "a.nii.gz"),
        image_info("pre", tmp_path / "b.nii.gz"),
        image_info("post", tmp_path / "c.nii.gz"),
    ]
    with pytest.raises(ValueError, match="Duplicate pairing key"):
        loader.organize_paired_data(images, "session", "pre", "post")
