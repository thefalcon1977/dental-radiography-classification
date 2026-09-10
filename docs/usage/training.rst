Training
========

Train a 3-class DenseNet121 on segmented dental patches.

Dataset layout
--------------

.. code-block:: text

   segmented_dental_adiography/
   ├── train/{dentin,enamel,pulp}/
   ├── valid/{dentin,enamel,pulp}/
   └── test/{dentin,enamel,pulp}/

Run
---

.. code-block:: bash

   python main.py --train

Saves the best checkpoint by validation accuracy and writes:

* ``training_history.png`` — train/val loss and accuracy
* ``confusion_matrix.png`` — held-out training test-set matrix

Device preference: CUDA → MPS → CPU.

Logic: :func:`densnet.train_runner.run_training`.

Final hyperparameters
----------------------

Defaults in :func:`densnet.train_runner.run_training`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - Backbone
     - DenseNet121, ImageNet-pretrained, 3-class head
   * - Optimizer
     - AdamW (``weight_decay=1e-4``)
   * - Learning rate
     - ``1e-4`` backbone; ``1e-3`` classifier head
   * - Batch size
     - 16
   * - Epochs
     - 30 maximum
   * - Early stopping
     - 7 epochs with no improvement in **validation accuracy**
   * - LR scheduler
     - ``ReduceLROnPlateau`` on validation **loss** (factor 0.5, patience 3)
   * - Loss
     - Weighted ``CrossEntropyLoss`` (inverse class frequency)
   * - Sampler
     - ``WeightedRandomSampler`` on the training set
   * - Gradient clip
     - max norm 1.0

Train / validation split
------------------------

The trainer does **not** randomly split a pool of 540 (or any) images.
It reads three folders that are already split on disk.

Current image counts (PNG/JPEG under
``segmented_dental_adiography/``):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Split
     - Dentin
     - Enamel
     - Pulp
     - Total
   * - Train
     - 239
     - 197
     - 168
     - **604** (69.6% of train+valid)
   * - Validation
     - 88
     - 94
     - 82
     - **264** (30.4% of train+valid)
   * - Test
     - 80
     - 80
     - 81
     - **241**
   * - **All**
     - 407
     - 371
     - 331
     - **1109**

Train : valid among the 868 training-time images is about **70% : 30%**.
The 241-image ``test/`` split is held out of fitting. External evaluation
in ``image-testing/`` (75 images per class) is separate from this table.

Data augmentation
-----------------

Applied **only** on the train split
(:func:`densnet.transforms.train_transform`). Validation and test use
:func:`densnet.transforms.eval_transform` (resize 256, center crop 224,
ImageNet normalize) with **no** flip, rotation, or jitter.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Transform
     - Setting
   * - Resize
     - 256×256
   * - Random resized crop (zoom/crop)
     - 224×224, scale 0.8–1.0
   * - Horizontal flip
     - probability 0.5
   * - Vertical flip
     - probability 0.3
   * - Rotation
     - ±15 degrees
   * - Color jitter
     - brightness 0.3, contrast 0.3, saturation 0.3, hue 0.1
   * - Affine (translate / scale)
     - translate 10%, scale 0.9–1.1 (no extra rotation)
   * - Random erasing
     - probability 0.2, area 0.02–0.33
   * - Normalize
     - ImageNet mean ``[0.485, 0.456, 0.406]``, std ``[0.229, 0.224, 0.225]``

Training history
----------------

.. image:: /_static/training_history.png
   :alt: Training and validation loss/accuracy curves
   :width: 100%

Confusion matrix
----------------

.. image:: /_static/confusion_matrix.png
   :alt: Confusion matrix on the training test set
   :width: 70%
