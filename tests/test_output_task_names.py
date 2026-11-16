import re
from pathlib import Path

import pytest

from statcraft.pipeline import StatCraftPipeline


def prefix(tasks):
    pipeline = object.__new__(StatCraftPipeline)
    pipeline._images = [
        {"path": Path(f"sub-{index:02}_" + (f"task-{task}_" if task else "")
                      + "space-MNI_stat.nii.gz")}
        for index, task in enumerate(tasks)
    ]
    return pipeline._generate_bids_prefix()


def test_composite_tasks_are_sorted_deduplicated_and_order_independent():
    actual = prefix(["task2", "task1", "task2"])
    assert re.fullmatch(r"task-task1task2H[0-9a-f]{12}_space-MNI", actual)
    assert actual == prefix(["task1", "task2"])
    assert actual != prefix(["task3", "task4"])


def test_hash_distinguishes_ambiguous_concatenations():
    assert prefix(["a", "bc"]) != prefix(["ab", "c"])


@pytest.mark.parametrize(
    ("tasks", "expected"),
    [([], "group"), (["rest", "rest"], "task-rest_space-MNI"),
     ([None, None], "space-MNI"), (["rest", None], "task-rest_space-MNI")],
)
def test_single_or_absent_tasks_keep_existing_names(tasks, expected):
    assert prefix(tasks) == expected


def test_missing_task_in_mixed_set_is_reported(caplog):
    assert prefix(["task1", "task2", None]).startswith("task-task1task2H")
    assert "missing a task entity" in caplog.text
