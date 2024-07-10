import torch
import rdkit
import rdkit.Chem as Chem
from .fast_jtnn import Vocab, MolTree, JTPropVAE
import pandas as pd
import numpy as np
import random, os
from tqdm.auto import tqdm
tqdm.pandas()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # Seed for DataLoader, add worker_init_fn=seed_worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_vocab_set(path: str):
    vocab = [x.strip("\r\n ") for x in open(path)] 
    vocab_set = set(vocab)
    return vocab_set

def get_vocab(path: str):
    vocab = [x.strip("\r\n ") for x in open(path)]
    vocab = Vocab(vocab)
    return vocab

def check_vocab(smiles: str, vocab_set: set):
    cset = set()
    mol = MolTree(smiles)
    for c in mol.nodes:
        cset.add(c.smiles)
    return cset.issubset(vocab_set)

def df_vocab_check(data: pd.DataFrame, smiles_col: str, vocab_set: set, drop: bool = False):
    print('--- DATAFRAME VAE VOCAB CHECK ---')
    temp_df = data.copy()
    print('Data points: %d\n'%(temp_df.shape[0]))
    print('Checking...')
    temp_df['vocab_checked'] = temp_df[smiles_col].progress_apply(lambda x: check_vocab(x, vocab_set))
    print('Checked.\n')
    if drop:
        print('Filtering...')
        temp_df = temp_df[temp_df['vocab_checked'] == True]
        temp_df.reset_index(drop=True, inplace = True)
        print('Final data points: %d'%(temp_df.shape[0]))
    return temp_df

def JTVAE_model_load(model_path: str, vocab_path: str, hidden_size: int = 450, latent_size: int = 56, depthT: int = 20, depthG: int = 3, device: str = 'cpu'):
    print('--- LOADING VAE MODEL ---')
    print('Model: %s\nVocab: %s\nHidden size: %d\nLatent size: %d\nDepthT: %d\nDepthG: %d\nDevice: %s\n'%(model_path, vocab_path, hidden_size, latent_size, depthT, depthG, device))
    vocab = get_vocab(vocab_path)
    model = JTPropVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(torch.device(device))
    print('FINISHED.\n')
    return model
