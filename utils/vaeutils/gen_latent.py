# %env CUDA_VISIBLE_DEVICES=''

import torch
import torch.nn as nn

import rdkit
# from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from rdkit.Chem import RDConfig
import argparse

from joblib import delayed,Parallel

import numpy as np  
import pandas as pd
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.append('%s/fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from fast_jtnn import *
import networkx as nx
from tqdm.auto import tqdm

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--isomeric', type=bool, default=True)
parser.add_argument('--savedir', required=True)
parser.add_argument('--pchembl', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--n_cpus', type=int, default=20)
# parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]=""
def penalized_logp_standard(mol):

    logP_mean = 2.4399606244103639873799239
    logP_std = 0.9293197802518905481505840
    SA_mean = -2.4485512208785431553792478
    SA_std = 0.4603110476923852334429910
    cycle_mean = -0.0307270378623088931402396
    cycle_std = 0.2163675785228087178335699

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    # print(logP_mean)

    standardized_log_p = (log_p - logP_mean) / logP_std
    standardized_SA = (SA - SA_mean) / SA_std
    standardized_cycle = (cycle_score - cycle_mean) / cycle_std
    return log_p,SA,cycle_score,standardized_log_p + standardized_SA + standardized_cycle

with open(args.data, 'r') as f:
    smiles = [line.rstrip('\n') for line in f]
if args.pchembl != '':
    pchembl = np.loadtxt(args.pchembl)
    assert pchembl.shape[0] == len(smiles)


vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

batch_size = 100
model = JTPropVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
if args.device == 'cpu':
    model.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(args.model))
    model = model.cuda()

smiles_rdkit = []
logP_values = []
SA_scores = []
cycle_scores = []
targets = []
n = len(smiles)

for i in range(n):
    mol = MolFromSmiles(smiles[i])
    smiles_rdkit.append(MolToSmiles(mol, isomericSmiles=args.isomeric))
    logp,sa,cycle,pen_p = penalized_logp_standard(mol)
    logP_values.append(logp)
    SA_scores.append(sa)
    cycle_scores.append(cycle)
    targets.append(pen_p)

logP_values = np.array(logP_values)
SA_scores = np.array(SA_scores)
cycle_scores = np.array(cycle_scores)
targets = np.array(targets)

latent_points = []
for i in tqdm(range(0, len(smiles), batch_size)):
    batch = smiles[i:i+batch_size]
    mol_vec = model.encode_and_samples_from_smiles(batch)
    latent_points.append(mol_vec.data.cpu().numpy())

if not os.path.isdir(args.savedir):
    os.mkdir(args.savedir)
latent_points = np.vstack(latent_points)

np.savetxt(args.savedir+'/latent_features.txt', latent_points)
np.savetxt(args.savedir+'/targets.txt', targets)
np.savetxt(args.savedir+'/logP_values.txt', np.array(logP_values))
np.savetxt(args.savedir+'/SA_scores.txt', np.array(SA_scores))
np.savetxt(args.savedir+'/cycle_scores.txt', np.array(cycle_scores))

df = pd.DataFrame({"rdkit_SMILES":smiles_rdkit})
df = pd.concat([df,pd.DataFrame({"pen_log_p":targets})],axis=1)
if args.pchembl != '':
    df = pd.concat([df,pd.DataFrame({"pChEMBL":pchembl})],axis=1)

df = pd.concat([df,pd.DataFrame(latent_points, columns=['latent_%d'%(i) for i in range(args.latent_size)])],axis=1)

df.to_csv(args.savedir+'/latent_features.csv')
