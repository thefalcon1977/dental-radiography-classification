Detection
=========

Sliding-window scan over a full radiograph with colored boxes:

* Red = dentin
* Green = enamel
* Blue = pulp

Defaults
--------

* Checkpoint: ``slm/resolution_best_densenet_model.pth``
* Patch size: ``224``
* Stride: ``56``
* Confidence threshold: ``0.90``
* NMS IoU threshold: ``0.35``

Run
---

.. code-block:: bash

   python main.py --detect
   # or: python main.py --detect --image path/to/xray.png

Each detection is ``(x1, y1, x2, y2, class_name, confidence)``.
