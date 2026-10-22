
PlantVillage
============

Overview
--------

PlantVillage is an image classification dataset of plant leaves, covering
healthy leaves and a range of plant diseases across 38 categories.

This implementation uses the color images from the original PlantVillage
repository. The dataset contains 54,305 images.

.. image:: teasers/plant_village_teaser.png
   :alt: PlantVillage leaf images from 12 example classes, labeled by plant and health condition
   :align: center
   :width: 100%

.. image:: experiments/plantvillage_intel/splits.png
   :alt: Exact train, validation and test image counts for both datasets
   :width: 100%

Dataset Information
-------------------

.. list-table::
   :header-rows: 1

   * - Property
     - Value
   * - Task
     - Image classification
   * - Number of classes
     - 38
   * - Total images
     - 54,305
   * - Training images
     - 43,429
   * - Validation images
     - 5,428
   * - Test images
     - 5,448
   * - Image format
     - RGB color images

Splits
------

The original color-image directory does not provide the train, validation,
and test partitions used by this builder.

Within each class, full ZIP-member paths are sorted by
``(SHA256("42:" + path), path)``. For a class of size N, the first ``floor(0.8N)``
paths form train, the next paths through ``floor(0.9N)`` form validation, and the
remainder form test. Output paths are sorted lexicographically. Split seed 42 is
independent of the model-training seeds. Archive ordering does not affect splits;
changing a repository prefix in a member path can affect assignments.

The downloaded snapshot has 54,305 color images, although the upstream overview
quotes 54,306. Counts here refer to the actual selected archive members.

The split operates at the image level. Images of the same physical leaf
may therefore occur in different partitions. This limitation should be
considered when interpreting evaluation results.

Usage
-----

.. code-block:: python

    from stable_datasets.images import PlantVillage

    ds_train = PlantVillage(split="train")
    ds_valid = PlantVillage(split="validation")
    ds_test = PlantVillage(split="test")

    ds_all = PlantVillage(split=None)

    print(len(ds_train))
    print(len(ds_valid))
    print(len(ds_test))

Each example contains:

* ``image``: The plant-leaf image.
* ``label``: An integer class identifier from 0 to 37.

Source
------

* Original repository: https://github.com/spMohanty/PlantVillage-Dataset
* Download source: https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip

Class mapping
-------------

The following names define the stable zero-based class order:

.. code-block:: text

    0: Apple___Apple_scab
    1: Apple___Black_rot
    2: Apple___Cedar_apple_rust
    3: Apple___healthy
    4: Blueberry___healthy
    5: Cherry_(including_sour)___Powdery_mildew
    6: Cherry_(including_sour)___healthy
    7: Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
    8: Corn_(maize)___Common_rust_
    9: Corn_(maize)___Northern_Leaf_Blight
    10: Corn_(maize)___healthy
    11: Grape___Black_rot
    12: Grape___Esca_(Black_Measles)
    13: Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
    14: Grape___healthy
    15: Orange___Haunglongbing_(Citrus_greening)
    16: Peach___Bacterial_spot
    17: Peach___healthy
    18: Pepper,_bell___Bacterial_spot
    19: Pepper,_bell___healthy
    20: Potato___Early_blight
    21: Potato___Late_blight
    22: Potato___healthy
    23: Raspberry___healthy
    24: Soybean___healthy
    25: Squash___Powdery_mildew
    26: Strawberry___Leaf_scorch
    27: Strawberry___healthy
    28: Tomato___Bacterial_spot
    29: Tomato___Early_blight
    30: Tomato___Late_blight
    31: Tomato___Leaf_Mold
    32: Tomato___Septoria_leaf_spot
    33: Tomato___Spider_mites Two-spotted_spider_mite
    34: Tomato___Target_Spot
    35: Tomato___Tomato_Yellow_Leaf_Curl_Virus
    36: Tomato___Tomato_mosaic_virus
    37: Tomato___healthy

Download, cache and RGB behavior
--------------------------------

The builder downloads the GitHub ZIP once. It reads selected members directly;
no archive entries are extracted to the filesystem. The first load builds a
processed cache per split. Repeated loads reopen it without downloading.

RGB images retain their original encoded bytes. Other modes, including a source
RGBA PNG, are converted to RGB and encoded as lossless PNG (regardless of the
original filename extension). Version 1.0.1 uses a new cache fingerprint so older
non-RGB caches are not silently reused. Existing older caches are not deleted.

The defaults are ``~/.stable-datasets/downloads`` and
``~/.stable-datasets/processed``. Override them explicitly with
``download_dir=...`` and ``processed_cache_dir=...``. ``STABLE_DATASETS_CACHE_DIR``
can relocate the dataset cache root; the benchmark runner separately uses
``STABLE_DATASETS_ROOT`` for its data directory.

Attribution and usage terms
---------------------------

Attribute Mohanty, Hughes and Salathe (2016),
`Using Deep Learning for Image-Based Plant Disease Detection
<https://doi.org/10.3389/fpls.2016.01419>`_. The downloaded GitHub archive contains
no explicit LICENSE file. The authors' current
`Hugging Face card <https://huggingface.co/datasets/mohanty/PlantVillage>`_
declares CC BY-SA 3.0, while its loader comments that the license should be
verified. That card is not treated here as a definitive license grant for every
file in the GitHub snapshot. Check the source's applicable terms before
redistribution. No dataset photographs are redistributed in this documentation.

.. code-block:: bibtex

    @article{Mohanty2016PlantVillage,
      title={Using Deep Learning for Image-Based Plant Disease Detection},
      author={Mohanty, Sharada P. and Hughes, David P. and Salathe, Marcel},
      journal={Frontiers in Plant Science},
      volume={7}, year={2016}, doi={10.3389/fpls.2016.01419}
    }

Validation and reproduction
---------------------------

SimCLR ViT-S/16 trained for **10 epochs with two seeds**. Final validation
balanced linear top-1 is **73.19% / 71.28%**; k-NN top-1 is **69.16% / 66.87%**.
The two-seed descriptive means are 72.24 +/- 1.35% and 68.01 +/- 1.62%
(sample standard deviation). Seed 0 selected epoch 10 with a validation stopping
rule; seed 1 repeated the fixed endpoint. Both use a 50-epoch LR horizon.

.. image:: experiments/plantvillage_intel/accuracy.png
   :alt: Intel and PlantVillage validation learning curves
   :width: 100%

See :doc:`the complete experiment report <experiments/plantvillage_intel/REPORT>` for
exact commands, environment, losses, provenance and limitations.
