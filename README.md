# Transformer-based Efﬁcient Salient Instance Segmentation Networks with Orientative Query

Official implementation of TMM2022 "Transformer-based Efﬁcient Salient Instance Segmentation Networks with Orientative Query"

## Environment preparation

The code is tested on CUDA 10.1 and pytorch 1.8.0, specify the versions below to your desired ones.

```shell
conda create -n oqtr python=3.8 -y
conda activate oqtr
git clone https://github.com/ssecv/OQTR
cd OQTR
conda install -c pytorch torchvision
pip install -r requirements.txt
```

## Data preparation

Revise `build_sis` function in `datasets/coco.py`.

## Run model

```shell
Python visualize.py --input {INPUT_IMG} --output {OUTPUT_DIR}
```

Please replace {INPUT_IMG} to you input image path and {OUTPUT_DIR} to your output path.

## Resources

- [OQTR-R50](https://github.com/ssecv/OQTR/releases/download/v1.0.0/oqtr_r50.pth)
- SIS10K
  - [Baidu Disk](https://pan.baidu.com/s/1ZOQAj0Lhg1K4Vi3eS5Tw6w) Verification code: hust
  - [Google Disk]() Coming soon

## Citation
```BibTeX
@article{pei2022oqtr,
  title={Transformer-based Efficient Salient Instance Segmentation Networks with Orientative Query},
  author={Pei, Jialun and Cheng, Tianyang and Tang, He and Chen, Chuanbo},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledge

The project is based on 
[DETR](https://github.com/facebookresearch/detr) and 
[CPD](https://github.com/wuzhe71/CPD), 
thanks them for their great work!