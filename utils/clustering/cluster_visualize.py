import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

class cluster_scatter_plot:
    """""
    Input:
    - df: dataframe
        must have Molecules column and cluster index
    - no_cls: int
        number of cluster in scatter plot
    - mol_col: string
        name of column containg Molecules
    - cluster_col: string
        name of column containg cluster index
    - algo: string
        name of clustering algorithm
    Return:
    - scatter plot
    """""
    def __init__(self,data, no_cls, mol_col, cluster_col, algo, fp_type = 'RDK7',palette=sns.color_palette("husl", 8),
                 PCA = True, perplexity = 50.0):
        self.data = data
        self.no_cls = no_cls
        self.mol_col = mol_col
        self.cluster_col = cluster_col
        self.algo = algo
        assert fp_type in ['RDK5', 'RDK6', 'RDK7', 'ECFP2', 'ECFP4', 'ECFP6']
        self.fp_type = fp_type
        self.palette = palette
        self.PCA = PCA
        self.perplexity = perplexity
    
    def mol2fp(self, mol):
        if self.fp_type == 'RDK5':
            fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=2048, nBitsPerHash=2)
        elif self.fp_type == 'RDK6':
            fp = Chem.RDKFingerprint(mol, maxPath=6, fpSize=2048, nBitsPerHash=2)
        elif self.fp_type == 'RDK7':
            fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=4096, nBitsPerHash=2)
        elif self.fp_type == 'ECFP2':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits = 2048)
        elif self.fp_type == 'ECFP4':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = 2048)
        elif self.fp_type == 'ECFP6':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits = 4096)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def processing(self):
        self.df_cls = self.data[[self.mol_col, self.cluster_col]]
        idx = self.df_cls[self.df_cls[self.cluster_col] >= self.no_cls].index
        self.df_cls.drop(idx, axis = 0, inplace = True)
        self.df_cls.reset_index(drop = True, inplace = True)
        
        # fp
        self.df_cls["FPs"] = self.df_cls[self.mol_col].apply(self.mol2fp)
        X = np.stack(self.df_cls.FPs.values)
        X_df = pd.DataFrame(X)
        self.df_cls= pd.concat([self.df_cls, X_df], axis = 1).drop(["FPs", self.mol_col], axis =1)
        
        # tSNE
        if self.PCA:
            pca  = PCA(n_components=50)
            in_tsne = pca.fit_transform(self.df_cls.drop(self.cluster_col, axis = 1))
        else:
            in_tsne = self.df_cls.drop(self.cluster_col, axis = 1)
        tSNE=TSNE(n_components=2, random_state = 42, perplexity = self.perplexity)
        tSNE_result= tSNE.fit_transform(in_tsne)
        self.x=tSNE_result[:,0]
        self.y=tSNE_result[:,1]
        
    def visualize(self):
        self.processing()
        sns.set_theme(font_scale=1, context ='notebook',style='darkgrid',)
        
        self.df_cls['x']=self.x
        self.df_cls['y']=self.y
        self.df_cls['legend'] = self.df_cls[self.cluster_col].apply(lambda x: 'Cluster %d'%(int(x)+1))
        self.df_cls.sort_values(by=['Cluster'], inplace = True)

        fig =plt.figure(figsize=(10,8))
        fig =sns.scatterplot(x='x',y='y',hue = 'legend',palette=self.palette,
                             data=self.df_cls,legend='auto')
        fig.set_title(f'{self.algo}', fontsize = 24, weight = 'semibold')
        fig.set_xlabel("tSNE-1", fontsize=16)
        fig.set_ylabel("tSNE-2", fontsize=16)
        fig.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1)
        # fig.legend()
        plt.savefig(f"{self.algo}.png", dpi = 600,bbox_inches="tight")
        
        
class cluster_heat_map:
    """""
    Input:
    - cls_cps: list
        list of molecules, must have name (GetProp('_Name'))
    - radius: int (2)
        ECFP value
    - nBits: int (2048)
        ECFP value
    Return:
    - 
    """""
    def __init__(self,cls_cps, fp_type = 'RDK7'):
        self.cls_cps = cls_cps
        assert fp_type in ['RDK5', 'RDK6', 'RDK7', 'ECFP2', 'ECFP4', 'ECFP6']
        self.fp_type = fp_type
        sns.set(font_scale=2)
        self.fig = plt.figure(figsize = (20,20))
        self.processing()
        
    def processing(self):
        if self.fp_type == 'RDK5':
            fps = [Chem.RDKFingerprint(mol, maxPath=5, fpSize=2048, nBitsPerHash=2) for mol in self.cls_cps]
        elif self.fp_type == 'RDK6':
            fps = [Chem.RDKFingerprint(mol, maxPath=6, fpSize=2048, nBitsPerHash=2) for mol in self.cls_cps]
        elif self.fp_type == 'RDK7':
            fps = [Chem.RDKFingerprint(mol, maxPath=7, fpSize=4096, nBitsPerHash=2) for mol in self.cls_cps]
        elif self.fp_type == 'ECFP2':
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits = 2048) for mol in self.cls_cps]
        elif self.fp_type == 'ECFP4':
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = 2048) for mol in self.cls_cps]
        elif self.fp_type == 'ECFP6':
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits = 4096) for mol in self.cls_cps]
        size=len(self.cls_cps)
        hmap=np.empty(shape=(size,size))
        self.table=pd.DataFrame()
        for index, i in enumerate(fps):
            for jndex, j in enumerate(fps):
                similarity=DataStructs.FingerprintSimilarity(i,j, metric=DataStructs.TanimotoSimilarity)
                hmap[index,jndex]=similarity
                self.table.loc[self.cls_cps[index].GetProp('_Name'),self.cls_cps[jndex].GetProp('_Name')]=similarity
    
    def visualize_triangle(self): 
        
        corr = self.table.corr()
        mask = np.tril(np.ones_like(corr, dtype=bool))
            # generating the plot
        
        self.fig = sns.heatmap(self.table, annot = True, annot_kws={"fontsize":15}, center=0,
                    square=True,  linewidths=.1, cbar_kws={"shrink": .1}, mask = mask)

        plt.title('Heatmap of Tanimoto Similarities', fontsize = 25 )
        
        
    def visualize_square(self):
            # generating the plot
         
        self.fig = sns.heatmap(self.table, annot = True, annot_kws={"fontsize":10}, center=0,
                    square=True,  linewidths=.7, cbar_kws={"shrink": .5}, vmin = 0, vmax = 1)

        plt.title('Heatmap of Tanimoto Similarities', fontsize = 24) # title with fontsize 20
    
