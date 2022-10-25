#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n movinet-test python=3.10

conda activate movinet-test

pip install av

wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py

wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
pip install git+https://github.com/Atze00/MoViNet-pytorch.git

mkdir -p video_data test_train_splits
unrar e test_train_splits.rar test_train_splits
rm test_train_splits.rar
unrar e hmdb51_org.rar
rm hmdb51_org.rar
mv *.rar video_data

python - <<'EOF'
import os
for files in os.listdir('video_data'):
    foldername = files.split('.')[0]
    os.system("mkdir -p video_data/" + foldername)
    os.system("unrar e video_data/"+ files + " video_data/"+foldername)
EOF

rm video_data/*.rar


wget -O a0_hmdb51_no_causal.pth https://cvhci.anthropomatik.kit.edu/~dschneider/checkpoints/hmdb51/movinet/a0/a0_hmdb51_no_causal.pth
