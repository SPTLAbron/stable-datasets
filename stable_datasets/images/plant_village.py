import hashlib
import io
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath

from PIL import Image as PILImage

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


class PlantVillage(BaseDatasetBuilder):
    VERSION = Version("1.0.1")

    SOURCE = DatasetSource(
        homepage="https://github.com/spMohanty/PlantVillage-Dataset",
        citation="""@article{Mohanty2016PlantVillage,
        title={Using Deep Learning for Image-Based Plant Disease Detection},
        author={Mohanty, Sharada P. and Hughes, David P. and Salathe, Marcel},
        journal={Frontiers in Plant Science},
        volume={7},
        year={2016},
        doi={10.3389/fpls.2016.01419}
        }""",
        assets={
            "images": DownloadInfo(
                url=("https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip")
            ),
        },
    )

    SPLIT_SEED = 42

    @staticmethod
    def _labels():
        return [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ]

    def _info(self):
        return DatasetInfo(
            description=(
                "PlantVillage original RGB leaf images in 38 "
                "plant-health categories. Deterministic 80/10/10 "
                "class-stratified image splits; not leaf-group-aware."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
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
        data_path = paths[0]

        return [
            SplitGenerator(
                name=split,
                gen_kwargs={"data_path": data_path, "split": split},
            )
            for split in self._candidate_splits()
        ]

    @classmethod
    def _image_members(cls, archive):
        groups = defaultdict(list)
        valid_labels = set(cls._labels())

        for name in archive.namelist():
            path = PurePosixPath(name)
            parts = path.parts

            if len(parts) < 4:
                continue

            # Accept either raw/color/... or a repository prefix.
            matches = [index for index in range(len(parts) - 2) if parts[index : index + 2] == ("raw", "color")]
            if not matches:
                continue

            index = matches[0]
            if len(parts) != index + 4:
                continue

            class_name = parts[index + 2]
            if class_name not in valid_labels:
                continue

            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            if name.endswith("/"):
                continue

            groups[class_name].append(name)

        return groups

    @classmethod
    def _split_members(cls, archive):
        groups = cls._image_members(archive)
        result = {
            Split.TRAIN: [],
            Split.VALIDATION: [],
            Split.TEST: [],
        }

        for class_name in sorted(groups):
            members = sorted(
                groups[class_name],
                key=lambda name: (
                    hashlib.sha256(f"{cls.SPLIT_SEED}:{name}".encode()).digest(),
                    name,
                ),
            )

            count = len(members)
            train_end = int(count * 0.8)
            validation_end = int(count * 0.9)

            result[Split.TRAIN].extend(members[:train_end])
            result[Split.VALIDATION].extend(members[train_end:validation_end])
            result[Split.TEST].extend(members[validation_end:])

        for split in result:
            result[split].sort()

        return result

    def _generate_examples(self, data_path, split):
        if split not in self._candidate_splits():
            raise ValueError(f"Unknown PlantVillage split: {split}")

        with zipfile.ZipFile(data_path) as archive:
            groups = self._split_members(archive)
            selected = groups[split]
            if not selected:
                raise ValueError(f"No PlantVillage color images found for split {split!r}. Check the archive layout.")

            label_to_id = {name: index for index, name in enumerate(self._labels())}

            for name in selected:
                class_name = PurePosixPath(name).parent.name
                image_bytes = archive.read(name)
                with PILImage.open(io.BytesIO(image_bytes)) as image:
                    if image.mode != "RGB":
                        buffer = io.BytesIO()
                        image.convert("RGB").save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                yield (
                    name,
                    {
                        "image": image_bytes,
                        "label": label_to_id[class_name],
                    },
                )
