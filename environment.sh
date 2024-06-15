conda update conda
### make sure you have your CUDA PATHs in .bashrc
### export PATH=/usr/local/cuda/bin:$PATH
### export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
### Otherwise, install mvcc from other routines: see https://mmcv.readthedocs.io/en/latest/get_started/installation.html
conda env create -f environment_segformer.yml
conda activate segformer
pip install -e . --user

