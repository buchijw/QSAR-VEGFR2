#!/bin/bash
conda install -c conda-forge numpy pandas matplotlib scikit-learn -y

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

conda install -c conda-forge scikit-posthocs ipykernel ipywidgets rdkit py-xgboost catboost imbalanced-learn networkx mhfp tqdm umap-learn -y

pip install argparse

pip install git+https://github.com/iwatobipen/CATS2D.git

mkdir -p built_packages && cd built_packages

git clone https://github.com/reymond-group/tmap.git

cd tmap

export LIBOGDF_INSTALL_PATH="$(pwd)/libOGDF"

mkdir -p $LIBOGDF_INSTALL_PATH

cd ..

git clone https://github.com/ogdf/ogdf.git

cd ogdf

mkdir -p build && cd build

CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$LIBOGDF_INSTALL_PATH \
        -DBUILD_SHARED_LIBS=ON

make -j10

sudo make install

cd ../..

pip install ogdf-python

cd tmap

CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ pip install .

cd ..

git clone https://github.com/guyrosin/mol2vec -b gensim_v4

cd mol2vec

pip install .

cd ../..

pip install git+https://github.com/reymond-group/map4@v1.0

pip install mordredcommunity[full]

conda install -c conda-forge jupyterlab python-kaleido -y