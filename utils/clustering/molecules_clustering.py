
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from rdkit.ML.Cluster import Butina
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm.auto import tqdm

from sklearn.manifold import TSNE

tqdm.pandas()
sns.set()
sns.set(rc={'figure.figsize': (12, 8)})
sns.set_style('darkgrid')



class Butina_clustering:
    """""
    Input:
    - df: dataframe
        must have Smiles column, Molecules column, ID column and bioactive columns
    - smiles_col: string
        name of smile column
    - mol_col: string
        name of mol column
    - active_col: string
        name of bioactive column ~ pIC50
    - mw: int => recomend 600
        molecular weight cutoff, value above cutoff will be removed
    - thresh: int => recomend 7
        bioactive cutoff, values above cutoff will be selected as active compounds
        in case of binary value, thresh must be 0.5  
    - dis_cutoff: float => recommend 0.7 for pharmacophore; 0.6 for docking
        dissimilarity, opposite to Tanimoto similarity cutoff
    - cps: float
        minimum number of compounds in cluster to select cluster center
    - radius: int (2)
        ECFP value
    - nBits: int (2048)
        ECFP value
    
    Return:
    - active_set: dataframe
        diverse active molecules selected, contaning Smiles, Molecules, ID and bioactive columns
    - cluster_centers: list
        list of active molecules selected
    - df_active: dataframe
        all active molecules with cluster index
    """""
    def __init__(self, df,ID, smiles_col, active_col, mol_col = 'ROMol', activity_thresh = 7, 
                 fp_type = 'RDK7', dis_cutoff = 0.7, cps = 5):
        self.data = df
        self.ID = ID
        self.smiles_col = smiles_col
        self.active_col = active_col
        self.mol_col = mol_col
        self.thresh = activity_thresh
        assert fp_type in ['RDK5', 'RDK6', 'RDK7', 'ECFP2', 'ECFP4', 'ECFP6']
        self.fp_type = fp_type
        self.cutoff = dis_cutoff
        self.cps = cps
    
  
    def create_cps(self, df):
        compounds = []
        for _, chembl_id, mol in df[[self.ID, self.mol_col]].itertuples():
            mol.SetProp('_Name',chembl_id)
            compounds.append(mol)
        if self.fp_type == 'RDK5':
            fingerprints = [Chem.RDKFingerprint(mol, maxPath=5, fpSize=2048, nBitsPerHash=2) for mol in compounds]
        elif self.fp_type == 'RDK6':
            fingerprints = [Chem.RDKFingerprint(mol, maxPath=6, fpSize=2048, nBitsPerHash=2) for mol in compounds]
        elif self.fp_type == 'RDK7':
            fingerprints = [Chem.RDKFingerprint(mol, maxPath=7, fpSize=4096, nBitsPerHash=2) for mol in compounds]
        elif self.fp_type == 'ECFP2':
            fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits = 2048) for mol in compounds]
        elif self.fp_type == 'ECFP4':
            fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = 2048) for mol in compounds]
        elif self.fp_type == 'ECFP6':
            fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits = 4096) for mol in compounds]
        return compounds, fingerprints
        
    
    @staticmethod
    def tanimoto_distance_matrix(fp_list):
        """Calculate distance matrix for fingerprint list"""
        dissimilarity_matrix = []
        # Notice how we are deliberately skipping the first and last items in the list
        # because we don't need to compare them against themselves
        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix
    
    @staticmethod
    def cluster_fingerprints(fingerprints, cutoff=0.2):
        """Cluster fingerprints
        Parameters:
            fingerprints
            cutoff: threshold for the clustering
        """
        # Calculate Tanimoto distance matrix
        distance_matrix = Butina_clustering.tanimoto_distance_matrix(fingerprints)
        # Now cluster the data with the implemented Butina algorithm:
        clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters
    
    @staticmethod
    def cutoff_analysis(fingerprints, start, end, step):
        for cutoff in np.arange(start, end, step):
            clusters = Butina_clustering.cluster_fingerprints(fingerprints, cutoff=cutoff)
            plt.close()
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.set_title(f"Threshold: {cutoff:3.2f}")
            ax.set_xlabel("Cluster index")
            ax.set_ylabel("Number of molecules")
            ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters])
            plt.show()
            # display(fig)
    
    def active_clustering(self):
        self.df_active = self.data[self.data[self.active_col] >= self.thresh]
        
        self.cls_data = self.df_active.copy().reset_index(drop = True)
        self.compounds,  self.fingerprints = self.create_cps(df = self.cls_data)

        clusters = Butina_clustering.cluster_fingerprints(self.fingerprints, cutoff=self.cutoff)

        # Give a short report about the numbers of clusters and their sizes
    
        num_single = sum(1 for c in clusters if len(c) == 1)
        num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
        num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
        num_clust_g50 = sum(1 for c in clusters if len(c) > 50)
        num_clust_g75 = sum(1 for c in clusters if len(c) > 75)

        print("total # clusters: ", len(clusters))
  
        print("# singletons: ", num_single)
        print("# clusters with >5 compounds: ", num_clust_g5)
        print("# clusters with >25 compounds: ", num_clust_g25)
        print("# clusters with >50 compounds: ", num_clust_g50)
        print("# clusters with >75 compounds: ", num_clust_g75)
        return clusters
        
    def data_processing(self):
            
            #self.df_all, self.df_active = self.filter_data()
            
            
        clusters = self.active_clustering()
        
       
        a = [c for c in clusters] # n = least compounds in cluster
        self.all_center_idx = [c[0] for c in a]
        a = [c for c in clusters if len(c) > self.cps] # n = least compounds in cluster
        self.center_idx = [c[0] for c in a]
        cluster_centers = [self.compounds[c[0]] for c in a]
        print("cluster_centers",len(cluster_centers))
       
        # active
        center_id = []
        for i in cluster_centers:
            center_id.append(i.GetProp('_Name'))
        
        idx =[]
        for key, value in enumerate(self.cls_data[self.ID]):
            if value in center_id:
                idx.append(key)
       
        active_set = self.cls_data.iloc[idx,:]
        
        #mark cluster
        self.cls_data['Cluster'] = np.zeros(len(self.cls_data))
        cls_df = pd.DataFrame(clusters).T
        for i in cls_df.columns:
            idx = cls_df.iloc[:,i].dropna().values
            self.cls_data.loc[idx, 'Cluster'] = i
    
        score = silhouette_score(self.fingerprints,self.cls_data['Cluster'], random_state= 42)
        print('Silhouette Score:', score)
        return active_set, cluster_centers, self.cls_data, score
    

