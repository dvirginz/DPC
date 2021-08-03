## Get SMAL
* Download smal https://smal.is.tue.mpg.de/downloads
* Download smpl https://smpl.is.tue.mpg.de/downloads
* make init file in datasets and smal
* create a python 2 env
* conda create -n smpl python=2.7 --yes
* pip install numpy chumpy plyfile tqdm plotly xvfbwrapper matplotlib psutil requests trimesh tqdm
* conda install -c plotly plotly-orca --yes
* pip install opencv-python==4.2.0.32
*   In ~/.bashrc
    SMPL_LOCATION=~/repos/DG2N/data/datasets/smal/SMPL_python_v.1.0.0/smpl
    export PYTHONPATH=$PYTHONPATH:$SMPL_LOCATION
* pythonrepos/DG2N/models/sub_models/3D-CODED/data/generate_data_animals.py

