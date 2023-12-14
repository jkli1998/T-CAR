## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python <= 3.8 (Mine 3.7)
- PyTorch >= 1.7 (Mine 1.9.1 (CUDA 11.1))
- torchvision >= 0.9 (Mine 0.10.1 (CUDA 11.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name tcar python=3.7
conda activate tcar

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install h5py
pip install scipy

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides ipdb
pip install scikit-learn

# follow PyTorch installation in https://pytorch.org/get-started/locally/
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex

# WARNING if you use older Versions of Pytorch (anything below 1.7), you will need a hard reset,
# as the newer version of apex does require newer pytorch versions. Ignore the hard reset otherwise.
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac

python setup.py install --cuda_ext --cpp_ext


# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/jkli1998/TCAR.git
cd TCAR

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR


```