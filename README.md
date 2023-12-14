# Zero-Shot Scene Graph Generation via Triplet Calibration and Reduction in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/dongxingning/SHA_GCL_for_SGG/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.9.1-%237732a8)

This repository contains the code for our paper [Zero-Shot Scene Graph Generation via Triplet Calibration and Reduction](https://arxiv.org/abs/2309.03542), which has been accepted by TOMM.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions, the recommended configuration is cuda-11.1 & pytorch-1.9.1.  

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

### Issues about the Zero-Shot Test File

The `zeroshot_triplet.pytorch` in  [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) was generated with the requirement of entity overlap. Unfortunately, the training data of current SGG methods do not restrict the entity overlap. Thus some of the triplets that appear in the training set will also appear in `zeroshot_triplet.pytorch`. We regenerated the `zeroshot_triplet_new.pytorch` to avoid some triplets being mistakenly treated as unseen triplets. For more details, please refer to `zs_check.ipynb`.

## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxT9s3JwIpoGz4cA?e=usU6TR).

## Perform training on Scene Graph Generation

### Set the dataset path

First, please organize all the files like this:
```bash
datasets
  |-- vg
    |--pretrained_faster_rcnn
      |--model_final.pth     
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    
```

### Choose a task

To comprehensively evaluate the performance, we follow three conventional tasks: 1) **Predicate Classification (PredCls)** predicts the relationships of all the pairwise objects by employing the given ground-truth bounding boxes and classes; 2) **Scene Graph Classification (SGCls)** predicts the objects classes and their pairwise relationships by employing the given ground-truth object bounding boxes; and 3) **Scene Graph Detection (SGDet)** detects all the objects in an image, and predicts their bounding boxes, classes, and pairwise relationships.

For **Predicate Classification (PredCls)**, you need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Examples of the Training Command
Training Example : (VG, PredCls)
```bash
# first train the triplet prune model
python prune_train_net.py

# then train the SGG model
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10006 --nproc_per_node=1 \
        tools/relation_train_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        SOLVER.PRE_VAL False \
        MODEL.ROI_RELATION_HEAD.LAMBDA_ 0.01 \
        MODEL.ROI_RELATION_HEAD.PRUNE_RATE 0.85 \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TCARPredictor \
        SOLVER.IMS_PER_BATCH 14 \
        TEST.IMS_PER_BATCH 1 \
        DTYPE "float16" \
        SOLVER.MAX_ITER 16000 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
        SOLVER.STEPS "(10000, 16000)" \
        SOLVER.VAL_PERIOD 20000 \
        SOLVER.CHECKPOINT_PERIOD 16000 \
        GLOVE_DIR ./datasets/vg/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ./datasets/vg/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./checkpoints/TCAR-predcls
```

## Evaluation

You can download our training model (TCAR PredCls) from [this link]([model_0016000.pth](https://1drv.ms/u/s!ArKjY2KidZWMjllNo_dDESSb5J_K?e=lV71oJ)). You can evaluate it by running the following command.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10006 --nproc_per_node=1 \
        tools/relation_test_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        SOLVER.PRE_VAL False \
        MODEL.ROI_RELATION_HEAD.LAMBDA_ 0.01 \
        MODEL.ROI_RELATION_HEAD.PRUNE_RATE 0.85 \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TCARPredictor \
        SOLVER.IMS_PER_BATCH 14 \
        TEST.IMS_PER_BATCH 1 \
        DTYPE "float16" \
        SOLVER.MAX_ITER 16000 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
        SOLVER.STEPS "(10000, 16000)" \
        SOLVER.VAL_PERIOD 20000 \
        SOLVER.CHECKPOINT_PERIOD 16000 \
        GLOVE_DIR ./datasets/vg/glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ./datasets/vg/pretrained_faster_rcnn/model_final.pth \
        OUTPUT_DIR ./checkpoints/TCAR-predcls
```

## Citation

```bash
@article{li2023zero,
  title={Zero-Shot Scene Graph Generation via Triplet Calibration and Reduction},
  author={Li, Jiankai and Wang, Yunhong and Li, Weixin},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2023},
  publisher={ACM New York, NY}
}
```

## Acknowledgment

Our code is on top of [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), we sincerely thank them for their well-designed codebase.
