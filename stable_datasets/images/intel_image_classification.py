import hashlib
import zipfile
from pathlib import PurePosixPath

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class IntelImageClassification(BaseDatasetBuilder):
    VERSION = Version("1.0.0")
    SPLIT_SEED = 42
    VALIDATION_FRACTION = 0.1

    CLASS_NAMES = (
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    )

    SOURCE = DatasetSource(
        homepage="https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
        citation=(
            "Intel Image Classification dataset, originally released for the Intel Image Classification Challenge."
        ),
        assets={
            "images": DownloadInfo(
                url=(
                    "https://huggingface.co/datasets/"
                    "resolverkatla/Intel-Image-Classification/"
                    "resolve/main/intel-image-classification.zip"
                )
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Natural-scene image classification with six categories. "
                "The original labeled test set is preserved; a deterministic "
                "class-stratified validation split is created from training images."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=list(self.CLASS_NAMES)),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        source = self._source()
        paths = bulk_download(
            [source["assets"]["images"]],
            dest_folder=self._raw_download_dir,
        )
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={"data_path": paths[0], "split": split},
            )
            for split in self._candidate_splits()
        ]

    @classmethod
    def _image_members(cls, archive):
        groups = {
            "train": {label: [] for label in cls.CLASS_NAMES},
            "test": {label: [] for label in cls.CLASS_NAMES},
        }

        for name in archive.namelist():
            path = PurePosixPath(name)
            if name.endswith("/") or path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            parts = path.parts
            for index, part in enumerate(parts):
                if part not in {"seg_train", "seg_test"}:
                    continue

                subset = "train" if part == "seg_train" else "test"

                remainder = parts[index + 1 :]
                if remainder and remainder[0] == part:
                    remainder = remainder[1:]

                if len(remainder) != 2:
                    continue

                label = remainder[0]
                if label in cls.CLASS_NAMES:
                    groups[subset][label].append(name)
                break

        return groups

    @classmethod
    def _split_members(cls, archive):
        groups = cls._image_members(archive)
        result = {
            Split.TRAIN: [],
            Split.VALIDATION: [],
            Split.TEST: [],
        }

        for label in cls.CLASS_NAMES:
            train_images = sorted(
                groups["train"][label],
                key=lambda name: (
                    hashlib.sha256(f"{cls.SPLIT_SEED}:{name}".encode()).digest(),
                    name,
                ),
            )

            validation_count = int(len(train_images) * cls.VALIDATION_FRACTION)

            result[Split.VALIDATION].extend(train_images[:validation_count])
            result[Split.TRAIN].extend(train_images[validation_count:])
            result[Split.TEST].extend(groups["test"][label])

        for split in result:
            result[split].sort()

        return result

    def _generate_examples(self, data_path, split):
        if split not in self._candidate_splits():
            raise ValueError(f"Unknown Intel Image Classification split: {split}")

        with zipfile.ZipFile(data_path) as archive:
            selected = self._split_members(archive)[split]

            if not selected:
                raise ValueError(
                    f"No Intel Image Classification images found for split {split!r}. Check the archive layout."
                )

            label_to_id = {name: index for index, name in enumerate(self.CLASS_NAMES)}

            for name in selected:
                label_name = PurePosixPath(name).parent.name
                yield (
                    name,
                    {
                        "image": archive.read(name),
                        "label": label_to_id[label_name],
                    },
                )
