import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools

from tqdm.auto import tqdm
tqdm.pandas()

class distance_maxtrix:
    """""
    Input:
    - data: dataframe
        must have Smiles column, ID column and bioactive columns
    - ID: string
        identification of molecules
    - mol_col: string
        name of column containing Molecules 
    - radius: int (2)
        ECFP value
    - nBits: int (2048)
        ECFP value
    Return:
    - df: dataframe
        containing cluster index columns
        
    """""
    def __init__(self,data, ID, mol_col,  dis_func, radius =2, nBits = 2048,fp=None,):
        self.data = data
        self.ID = ID
        self.mol_col = mol_col
        self.fp = fp
        self.dis_func = dis_func
        self.radius = radius
        self.nBits = nBits
        self.table = pd.DataFrame()
        
        
    def mol2fp(self):
        if self.fp is not None:
            self.list_mol = []
            self.list_fps = []
            for _, ID, mol, fp in self.data[[self.ID, self.smile_col, self.fp]].itertuples():
                mol.SetProp('_Name',ID)
                self.list_mol.append(mol)
                self.list_fps.append(fp)
        else:
            self.list_mol = []
            for _, ID, mol in self.data[[self.ID, self.mol_col]].itertuples(): 
                mol.SetProp('_Name',ID)
                self.list_mol.append(mol)
            self.list_fps= [AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits = self.nBits) 
                            for mol in self.list_mol]
        
        
    def euc_bit(self, a: np.array, b: np.array) -> float:
        """Compute Euclidean distance from bitstring.
        Parameters
        ----------
        a : array_like
            molecule A's features in bits.
        b : array_like
            molecules B's features in bits.
        Returns
        -------
        e_d : float
            Euclidean distance between molecule A and B.
        Notes
        -----
        Bajusz, D., Rácz, A., and Héberger, K.. (2015)
        Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?.
        Journal of Cheminformatics 7.
        """
        a_feat = np.count_nonzero(a)
        b_feat = np.count_nonzero(b)
        c = 0
        for idx, _ in enumerate(a):
            if a[idx] == b[idx] and a[idx] != 0:
                c += 1
        e_d = (a_feat + b_feat - (2 * c)) ** 0.5
        return e_d
    
    def calculate_distance_maxtrix(self):
        self.mol2fp()
        self.table = pd.DataFrame()
        self.hmap=np.empty(shape=(len(self.list_mol),len(self.list_mol)))
        for index, i in enumerate(self.list_fps):
            for jndex, j in enumerate(self.list_fps):
                if self.dis_func == 'Tanimoto':
                #similarity=euc_bit(i, j)
                    dissimilarity=1-DataStructs.FingerprintSimilarity(i,j, metric=DataStructs.TanimotoSimilarity)
                self.hmap[index,jndex]=dissimilarity
                self.table.loc[self.list_mol[index].GetProp('_Name'),self.list_mol[jndex].GetProp('_Name')]=dissimilarity
        return self.table
    
    


class diverse_subset:
    def __init__(self, data, cluster_col, ID, num_selected, mol_col, method ='MaxMin'):
        self.data = data
        self.cluster_col = cluster_col
        self.ID = ID
        self.num_selected = num_selected
        self.mol_col = mol_col
        self.method = method
        
    def MaxMin(self, distance_tab, num_selected):
        """
        Algorithm MinMax for selecting points from cluster.
        Parameters
        ----------
        distance_tab: pd.DataFrame
             Distance matrix for points.
        num_selected: int
            Number of molecules that need to be selected
        Returns
        -------
        selected: list
            List of ids of selected molecules
        """

        arr_dist = distance_tab.values


        # choosing initial point as the medoid
        selected = [np.argmin(np.sum(arr_dist, axis=0))]
        while len(selected) < num_selected:
            min_distances = np.min(arr_dist[selected], axis=0)
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        id_name = distance_tab.iloc[selected,selected].index
        return id_name
    
    def MaxSum(self, distance_tab, num_selected):
        """
        Algorithm MinMax for selecting points from cluster.
        Parameters
        ----------
        distance_tab: pd.DataFrame
             Distance matrix for points.
        num_selected: int
            Number of molecules that need to be selected
        Returns
        -------
        selected: list
            List of ids of selected molecules
        """

        arr_dist = distance_tab.values


        # choosing initial point as the medoid
        selected = [np.argmin(np.sum(arr_dist, axis=0))]
        while len(selected) < num_selected:
            sum_distances = np.sum(arr_dist[selected], axis=0)
            while True:
                new_id = np.argmax(sum_distances)
                if new_id in selected:
                        sum_distances[new_id] = 0
                else:
                    break
            selected.append(new_id)
        id_name = distance_tab.iloc[selected,selected].index
        return id_name
    
    def pick_subset(self):
        df = self.data.copy()
        number_cluster = len(np.unique(df[self.cluster_col]))
        self.num_selected_cls = int(self.num_selected/number_cluster)
        ID_name = []
        if self.method == 'MaxMin':
            for i in tqdm(range(number_cluster)):
                idx = df[df[self.cluster_col]==i].index
                cluster_df = df.iloc[idx,:].reset_index(drop=True)
                matrix = distance_maxtrix(data = cluster_df, ID=self.ID, mol_col=self.mol_col, dis_func='Tanimoto')
                table = matrix.calculate_distance_maxtrix()
                id_name = self.MaxMin(distance_tab=table, num_selected = self.num_selected_cls)
                ID_name.append(id_name)
        elif self.method == 'MaxSum':
            for i in tqdm(range(number_cluster)):
                idx = df[df[self.cluster_col]==i].index
                cluster_df = df.iloc[idx,:].reset_index(drop=True)
                matrix = distance_maxtrix(data = cluster_df, ID=self.ID, mol_col=self.mol_col, dis_func='Tanimoto')
                table = matrix.calculate_distance_maxtrix()
                id_name = self.MaxSum(distance_tab=table, num_selected = self.num_selected_cls)
                ID_name.append(id_name)


        ID_cls= []
        for i in ID_name:
            for j in i:
                ID_cls.append(j)

        idx =[]
        for key, value in enumerate(df[self.ID]):
            if value in ID_cls:
                idx.append(key)
        return df.iloc[idx,:]
