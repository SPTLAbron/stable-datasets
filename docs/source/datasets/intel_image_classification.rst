
Intel Image Classification
==========================

Overview
--------

Intel Image Classification is a natural-scene image classification
dataset containing photographs in six categories:

* buildings
* forest
* glacier
* mountain
* sea
* street

The dataset was originally published for an Intel image
classification challenge.

Dataset Information
-------------------

.. list-table::
   :header-rows: 1

   * - Property
     - Value
   * - Task
     - Image classification
   * - Number of classes
     - 6
   * - Original labeled training images
     - 14,034
   * - Original labeled test images
     - 3,000
   * - Image format
     - RGB images

Splits
------

The builder preserves the original labeled test set.

The original training set is divided into training and validation
partitions using deterministic, class-stratified image-path hashing.
Approximately 10 percent of each training class is assigned to
validation.

The original unlabeled prediction set is excluded because it does
not provide class labels for supervised evaluation.

Usage
-----

.. code-block:: python

    from stable_datasets.images import IntelImageClassification

    ds_train = IntelImageClassification(split="train")
    ds_valid = IntelImageClassification(split="validation")
    ds_test = IntelImageClassification(split="test")

    ds_all = IntelImageClassification(split=None)

    print(len(ds_train))
    print(len(ds_valid))
    print(len(ds_test))

Each example contains:

* ``image``: The natural-scene image.
* ``label``: An integer class identifier from 0 to 5.

Source
------

* Original dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
* Community-hosted archive: https://huggingface.co/datasets/resolverkatla/Intel-Image-Classification

