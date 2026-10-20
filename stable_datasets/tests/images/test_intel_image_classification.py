import io
import zipfile

import pytest
from PIL import Image

from stable_datasets.images.intel_image_classification import (
    IntelImageClassification,
)
from stable_datasets.splits import Split


def make_image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="green").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def intel_archive(tmp_path):
    archive_path = tmp_path / "intel.zip"
    image_bytes = make_image_bytes()

    with zipfile.ZipFile(archive_path, "w") as archive:
        for label in IntelImageClassification.CLASS_NAMES:
            for index in range(20):
                archive.writestr(
                    f"seg_train/seg_train/{label}/{index}.jpg",
                    image_bytes,
                )

            for index in range(3):
                archive.writestr(
                    f"seg_test/seg_test/{label}/{index}.jpg",
                    image_bytes,
                )

        # Prediction images have no class labels.
        archive.writestr(
            "seg_pred/seg_pred/unlabeled.jpg",
            image_bytes,
        )

    return archive_path


def make_builder():
    builder = object.__new__(IntelImageClassification)
    builder._dataset_info = builder._info()
    return builder


def test_class_names():
    assert len(IntelImageClassification.CLASS_NAMES) == 6
    assert len(set(IntelImageClassification.CLASS_NAMES)) == 6


def test_candidate_splits():
    assert make_builder()._candidate_splits() == [
        Split.TRAIN,
        Split.VALIDATION,
        Split.TEST,
    ]


def test_split_counts(intel_archive):
    with zipfile.ZipFile(intel_archive) as archive:
        splits = IntelImageClassification._split_members(archive)

    assert len(splits[Split.TRAIN]) == 108
    assert len(splits[Split.VALIDATION]) == 12
    assert len(splits[Split.TEST]) == 18


def test_splits_are_deterministic(intel_archive):
    with zipfile.ZipFile(intel_archive) as archive:
        first = IntelImageClassification._split_members(archive)
        second = IntelImageClassification._split_members(archive)

    assert first == second


def test_no_split_overlap(intel_archive):
    with zipfile.ZipFile(intel_archive) as archive:
        splits = IntelImageClassification._split_members(archive)

    train = set(splits[Split.TRAIN])
    validation = set(splits[Split.VALIDATION])
    test = set(splits[Split.TEST])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


@pytest.mark.parametrize(
    ("split", "expected_count"),
    [
        (Split.TRAIN, 108),
        (Split.VALIDATION, 12),
        (Split.TEST, 18),
    ],
)
def test_generate_examples(intel_archive, split, expected_count):
    examples = list(
        make_builder()._generate_examples(
            data_path=intel_archive,
            split=split,
        )
    )

    assert len(examples) == expected_count

    for key, example in examples:
        assert key.endswith(".jpg")
        assert 0 <= example["label"] < 6

        with Image.open(io.BytesIO(example["image"])) as image:
            assert image.size == (8, 8)


def test_unknown_split(intel_archive):
    with pytest.raises(ValueError, match="Unknown Intel"):
        list(
            make_builder()._generate_examples(
                data_path=intel_archive,
                split="invalid",
            )
        )


def test_empty_archive(tmp_path):
    archive_path = tmp_path / "empty.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "empty")

    with pytest.raises(ValueError, match="No Intel"):
        list(
            make_builder()._generate_examples(
                data_path=archive_path,
                split=Split.TRAIN,
            )
        )
