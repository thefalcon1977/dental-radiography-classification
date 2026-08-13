Evaluation
==========

Aggregate the three prediction CSVs and compute one-vs-rest metrics.

Formulas
--------

* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* Accuracy = (TP + TN) / (TP + TN + FP + FN)
* F1 = 2 × Precision × Recall / (Precision + Recall)

Safe division by zero returns ``0.0``.

Run
---

.. code-block:: bash

   python evaluate_test_predictions.py

Writes:

* ``test_predictions/evaluation_metrics.csv``
* ``test_predictions/evaluation_report.txt``

See the repository ``README.md`` for the latest external-test summary table.
