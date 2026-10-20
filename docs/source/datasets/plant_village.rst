
PlantVillage
============

Overview
--------

PlantVillage is an image classification dataset of plant leaves, covering
healthy leaves and a range of plant diseases across 38 categories.

This implementation uses the color images from the original PlantVillage
repository. The dataset contains 54,305 images.

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

The implementation generates deterministic, class-stratified partitions
using an 80/10/10 split with a fixed seed of 42.

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
