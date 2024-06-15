#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=
conda deactivate
conda activate segformer
python3 ./tools/test.py local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py segformer.b5.1024x1024.city.160k.pth | tee base.log
