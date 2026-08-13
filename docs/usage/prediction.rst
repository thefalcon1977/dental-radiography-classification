Batch prediction
================

Classify held-out patches under ``image-testing/`` and write CSVs under
``test_predictions/``.

Folders
-------

.. code-block:: text

   image-testing/
   ├── dentin_test/
   ├── enamel_test/
   └── pulp_test/

CSV columns
-----------

``file,class_name,target,target_label,pred_idx,pred_label,prob_positive``

``prob_positive`` is the softmax probability of that script’s **target** class
(P(dentin), P(enamel), or P(pulp)).

Run
---

.. code-block:: bash

   python predict_dentin_test.py
   python predict_enamel_test.py
   python predict_pulp_test.py

=======  ==============================  =========================================
Script   Input                           Output
=======  ==============================  =========================================
dentin   ``image-testing/dentin_test/``  ``test_predictions/dentin_test_predictions.csv``
enamel   ``image-testing/enamel_test/``  ``test_predictions/enamel_test_predictions.csv``
pulp     ``image-testing/pulp_test/``    ``test_predictions/pulp_test_predictions.csv``
=======  ==============================  =========================================
