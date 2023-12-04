# 3D Contour Generation based on Diffusion Probabilistic Models

## Description

This repository contains the code for our paper: U3D Contour Generation based on Diffusion Probabilistic Models

<div align="center">
<img src="https://github.com/djzgroup/ContourGeneration/tree/main/images/Pipeline.png" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/djzgroup/ContourGeneration/tree/main/images/VisualizationResults.png" width="70%" height="70%"><br><br>
</div>

## Environment setup

```
pip install torch==1.11.0+cu111 torchvision==0.12.0+cu111 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pointnet2_ops_lib/.
```

## Dataset

The training and testing data for correspondence is provided by [ABC](https://github.com/wangxiaogang866/PIE-NET/tree/master/main/train_data) and [ShapeNet](https://github.com/antao97/PointCloudDatasets)

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[Diffusion Probabilistic Models for 3D Point Cloud Generation](https://github.com/luost26/diffusionpoint-cloud),
[dgcnn](https://github.com/WangYueFt/dgcnn)
