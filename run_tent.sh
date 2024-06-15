export PYTHONPATH=
conda deactivate
conda activate segformer
bash ./tools/dist_tent.sh local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py segformer.b5.1024x1024.city.160k.pth 1 | tee tent.log
