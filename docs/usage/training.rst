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
