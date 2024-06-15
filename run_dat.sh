# source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=
conda deactivate
# bash environment.sh to install the env
conda activate segformer
bash ./tools/dist_dat.sh local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py segformer.b5.1024x1024.city.160k.pth 1 | tee dat.log
