#!/bin/bash
conda install -c conda-forge numpy pandas matplotlib scikit-learn -y

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install -c conda-forge scikit-posthocs ipykernel ipywidgets rdkit py-xgboost catboost imbalanced-learn networkx mhfp tqdm umap-learn -y

conda install -c kotori_y scopy -y

pip install argparse

pip install git+https://github.com/iwatobipen/CATS2D.git

mkdir -p built_packages && cd built_packages

git clone https://github.com/ogdf/ogdf.git

cd ogdf

mkdir -p build && cd build

CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON

make -j10

sudo make install

cd ../..

pip install ogdf-python

git clone https://github.com/reymond-group/tmap.git

cd tmap

CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install .

cd ..

git clone https://github.com/guyrosin/mol2vec -b gensim_v4

cd mol2vec

pip install .

cd ../..

pip install git+https://github.com/reymond-group/map4@v1.0

pip install mordredcommunity[full]

conda install -c conda-forge jupyterlab python-kaleido -y