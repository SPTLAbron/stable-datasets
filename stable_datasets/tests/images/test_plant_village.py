import io
import zipfile

import pytest
from PIL import Image

from stable_datasets.images.plant_village import PlantVillage
from stable_datasets.splits import Split


def make_image_bytes():
    image = Image.new("RGB", (8, 8), color=(30, 150, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def plant_village_archive(tmp_path):
    archive_path = tmp_path / "plantvillage.zip"
    image_bytes = make_image_bytes()
    labels = PlantVillage._labels()[:2]

    with zipfile.ZipFile(archive_path, "w") as archive:
        for label in labels:
            for index in range(20):
                archive.writestr(
                    f"PlantVillage-Dataset-master/raw/color/{label}/image_{index:03d}.jpg",
                    image_bytes,
                )

        archive.writestr(
            f"PlantVillage-Dataset-master/raw/grayscale/{labels[0]}/ignored.jpg",
            image_bytes,
        )

        archive.writestr(
            f"PlantVillage-Dataset-master/raw/color/{labels[0]}/notes.txt",
            "not an image",
        )

    return archive_path


def make_builder():
    builder = object.__new__(PlantVillage)
    builder._dataset_info = builder._info()
    return builder


def test_label_count():
    labels = PlantVillage._labels()
    assert len(labels) == 38
    assert len(set(labels)) == 38


def test_candidate_splits():
    builder = make_builder()
    assert builder._candidate_splits() == [
        Split.TRAIN,
        Split.VALIDATION,
        Split.TEST,
    ]


def test_split_sizes(plant_village_archive):
    with zipfile.ZipFile(plant_village_archive) as archive:
        splits = PlantVillage._split_members(archive)

    assert len(splits[Split.TRAIN]) == 32
    assert len(splits[Split.VALIDATION]) == 4
    assert len(splits[Split.TEST]) == 4


def test_splits_are_disjoint(plant_village_archive):
    with zipfile.ZipFile(plant_village_archive) as archive:
        splits = PlantVillage._split_members(archive)

    train = set(splits[Split.TRAIN])
    validation = set(splits[Split.VALIDATION])
    test = set(splits[Split.TEST])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(train | validation | test) == 40


def test_splits_are_deterministic(plant_village_archive):
    with zipfile.ZipFile(plant_village_archive) as archive:
        first = PlantVillage._split_members(archive)
        second = PlantVillage._split_members(archive)

    assert first == second


@pytest.mark.parametrize(
    "split",
    [Split.TRAIN, Split.VALIDATION, Split.TEST],
)
def test_generate_examples(plant_village_archive, split):
    builder = make_builder()

    examples = list(
        builder._generate_examples(
            data_path=plant_village_archive,
            split=split,
        )
    )

    assert examples

    for key, example in examples:
        assert key.endswith(".jpg")
        assert isinstance(example["image"], bytes)
        assert 0 <= example["label"] < 38

        with Image.open(io.BytesIO(example["image"])) as image:
            assert image.size == (8, 8)


def test_unknown_split_raises(plant_village_archive):
    builder = make_builder()

    with pytest.raises(ValueError, match="Unknown PlantVillage split"):
        list(
            builder._generate_examples(
                data_path=plant_village_archive,
                split="invalid",
            )
        )


def test_no_color_images_raises(tmp_path):
    archive_path = tmp_path / "empty.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "No images")

    builder = make_builder()

    with pytest.raises(ValueError, match="No PlantVillage color images"):
        list(
            builder._generate_examples(
                data_path=archive_path,
                split=Split.TRAIN,
            )
        )
