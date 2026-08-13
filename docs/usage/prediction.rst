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

   python main.py --predict all
   # or one class:
   python main.py --predict dentin
   python main.py --predict enamel
   python main.py --predict pulp

=======  ==============================  =========================================
Command  Input                           Output
=======  ==============================  =========================================
``--predict dentin``   ``image-testing/dentin_test/``  ``test_predictions/dentin_test_predictions.csv``
``--predict enamel``   ``image-testing/enamel_test/``  ``test_predictions/enamel_test_predictions.csv``
``--predict pulp``     ``image-testing/pulp_test/``    ``test_predictions/pulp_test_predictions.csv``
=======  ==============================  =========================================
