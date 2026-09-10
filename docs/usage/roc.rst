ROC and AUC
===========

Score held-out patches under ``image-testing/`` and compute **one-vs-rest
(OvR)** ROC curves and AUC for dentin, enamel, and pulp.

Each class is the positive class against the other two. The score is that
class's **softmax probability**, not the predicted label.

This command **re-runs inference** so it has all three class probabilities. It
does not read the :doc:`prediction` CSVs (those only store ``prob_positive``).

Uses the same :func:`densnet.transforms.eval_transform` preprocessing as
``--predict``. Default checkpoint:
``slm/resolution_best_densenet_model.pth``.

Run
---

.. code-block:: bash

   python main.py --roc

Logic: :func:`densnet.roc_auc.run_roc_auc`.

One-vs-rest
-----------

For class *c* (for example enamel):

* Positive samples: true label is *c*
* Negative samples: true label is one of the other two classes
* Score: ``P(c)`` from softmax

=======  ==========================
Class    Comparison
=======  ==========================
dentin   dentin vs enamel+pulp
enamel   enamel vs dentin+pulp
pulp     pulp vs dentin+enamel
=======  ==========================

Formulas
--------

* True positive rate (sensitivity) = TP / (TP + FN)
* False positive rate (1 − specificity) = FP / (FP + TN)
* AUC = area under the ROC curve (TPR vs FPR)
* Macro AUC (OvR) = unweighted mean of the three class AUCs
* Weighted AUC (OvR) = mean of class AUCs weighted by the number of positives

A chance classifier has AUC = 0.5 (the diagonal on the plot).

Outputs
--------

Writes under ``test_predictions/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Contents
   * - ``multiclass_probabilities.csv``
     - Per-image softmax for all three classes
   * - ``roc_curve_dentin.csv``
     - FPR, TPR, threshold for dentin vs rest
   * - ``roc_curve_enamel.csv``
     - FPR, TPR, threshold for enamel vs rest
   * - ``roc_curve_pulp.csv``
     - FPR, TPR, threshold for pulp vs rest
   * - ``one_vs_rest_auc_results.csv``
     - Per-class AUC (and sklearn check)
   * - ``roc_auc_summary.csv``
     - Macro / weighted OvR AUC, ``n_images``
   * - ``one_vs_rest_roc_curves.png``
     - ROC plot (one curve per class)

Probability CSV columns:

``file,true_idx,true_class,pred_idx,pred_label,prob_dentin,prob_enamel,prob_pulp,probability_sum``

``probability_sum`` is a sanity check (softmax should sum to 1).

See also
--------

* :doc:`evaluation` — Precision, Recall, Accuracy, F1 from ``--predict`` CSVs
* :doc:`prediction` — batch labels and ``prob_positive``
* :mod:`densnet.roc_auc` — public functions
