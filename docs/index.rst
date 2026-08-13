densNet documentation
=====================

Classify dental tissue in radiography images into **dentin**, **enamel**, and
**pulp** using DenseNet121.

Quick start
-----------

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python main.py --train

Classes (index order): ``0 = dentin``, ``1 = enamel``, ``2 = pulp``.

Default inference checkpoint: ``slm/resolution_best_densenet_model.pth``.

For Persian setup notes see ``README_FA.md`` in the repository root.
English narrative overview: ``README.md``.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/training
   usage/detection
   usage/prediction
   usage/evaluation

.. toctree::
   :maxdepth: 2
   :caption: API

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
