## Changes to MODEST
* `downstream/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py`
to support score filtering for pseudo ground-truth database used in data augmentation.

* `scripts/` and `generate_cluster_mask/` to support score filtering for pseudo-labels with and without static labels retention.  

* `downstream/OpenPCDet/pcdet/datasets/dataset.py` to support exceptions raised during training. 

* `downstream/OpenPCDet/tools/test.py` for fixed seed in evaluation.

* `downstream/OpenPCDet/tools/` to support evaluation of pseudo-labels against ground-truth annotations. 