#!/bin/bash
conda install nomkl -y

conda install -c conda-forge numpy pandas matplotlib scikit-learn -y

conda install pytorch torchvision torchaudio -c pytorch -y

conda install -c conda-forge scikit-posthocs ipykernel ipywidgets rdkit py-xgboost catboost imbalanced-learn networkx mhfp tqdm umap-learn -y

conda install -c kotori_y scopy -y

pip install argparse

pip install git+https://github.com/iwatobipen/CATS2D.git

mkdir -p built_packages && cd built_packages

git clone https://github.com/reymond-group/tmap.git

cd tmap

export LIBOGDF_INSTALL_PATH="$(pwd)/libOGDF"

mkdir -p $LIBOGDF_INSTALL_PATH

cd ./ogdf-conda/src && mkdir -p build && cd build

CC=/usr/local/opt/llvm/bin/clang CXX=/usr/local/opt/llvm/bin/clang++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$LIBOGDF_INSTALL_PATH \
        -DBUILD_SHARED_LIBS=ON

make -j8

make install

cd ../../..

CC=/usr/local/opt/llvm/bin/clang CXX=/usr/local/opt/llvm/bin/clang++ pip install .

cd ..

git clone https://github.com/guyrosin/mol2vec -b gensim_v4

cd mol2vec

pip install .

cd ../..

pip install git+https://github.com/reymond-group/map4@v1.0

pip install mordredcommunity[full] -y

conda install -c conda-forge jupyterlab python-kaleido -y

conda remove mkl mkl-service -y