# DAT: Distribution Aware Tuning

Official code for [Distribution-Aware Continual Test-Time Adaptation for Semantic Segmentation](https://arxiv.org/abs/2309.13604).

This repo contains detailed implementation for [teacher-student forward propagation](https://github.com/RochelleNi/DAT/blob/main/ours.py#L113), [uncertainty calculation](https://github.com/RochelleNi/DAT/blob/main/ours.py#L119), [pixel-level distribution shift evaluation](https://github.com/RochelleNi/DAT/blob/main/ours.py#L160) and [parameter selection and accumulation](https://github.com/RochelleNi/DAT/blob/main/ours.py#L176).

Training process can refer to [cotta](https://github.com/qinenergy/cotta) paper.

Experimental results of DAT model on Cityscapes_to_ACDC and SHIFT datasets can be found in [acdc](https://github.com/RochelleNi/DAT/blob/main/acdc.log) and [shift](https://github.com/RochelleNi/DAT/blob/main/shift.log).

## Citation
Please cite our work if you find it useful.

```bibtex
@misc{ni2024distributionaware,
      title={Distribution-Aware Continual Test-Time Adaptation for Semantic Segmentation}, 
      author={Jiayi Ni and Senqiao Yang and Ran Xu and Jiaming Liu and Xiaoqi Li and Wenyu Jiao and Zehui Chen and Yi Liu and Shanghang Zhang},
      year={2024}
}
```