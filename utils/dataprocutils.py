import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans
from plotutils import PCA_2D, PCA_3D
from scipy import stats
from tqdm.auto import tqdm
tqdm.pandas()

class OutliersHandle:
    def __init__(self, ):
        pass
    
    @staticmethod
    def DBSCAN_to_outliers(data: np.array, eps: float, min_samples: int):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = db.labels_
        outliers_idx = np.where(labels == -1)[0]
        return outliers_idx

    @classmethod
    def DBSCAN_remove(self, data: pd.DataFrame, repre: np.array, eps: float, min_samples: int, activity_col: str = None, verbose: bool = True):
        print('--- REMOVE OUTLIERS USING DBSCAN ---') if verbose else None
        temp_df = data.copy()
        print('Data points: %d'%(temp_df.shape[0])) if verbose else None
        print('Representation shape: ',repre.shape) if verbose else None
        print('Eps: %f'%(eps)) if verbose else None
        print('Min samples: %d'%(min_samples)) if verbose else None
        
        print('** Original representation with 2D PCA **') if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values
        else:
            c = None
        pca = PCA_2D(repre)
        fig = pca.plot(repre, c=c, cmap='crest', hull=True,
                                title='Latent space visualization - 2D PCA (Before)', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        print('** DBSCAN outlier removal **') if verbose else None
        outliers_idx = self.DBSCAN_to_outliers(repre, eps, min_samples)
        print('Number of outliers removed: %d'%(len(outliers_idx))) if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values[outliers_idx]
        else:
            c = None
        fig2 = pca.plot(repre[outliers_idx], c=c, cmap='crest', hull=False,
                                title='Outliers - 2D PCA', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        temp_df.drop(outliers_idx, inplace = True)
        temp_df.reset_index(drop=True, inplace = True)
        post_repre = np.delete(repre, outliers_idx, axis=0)
        
        print('Final data points: %d\n'%(temp_df.shape[0])) if verbose else None
        
        print('** Post-re Representation with 2D PCA **') if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values
        else:
            c = None
        fig3 = pca.plot(post_repre, c=c, cmap='crest', hull=True,
                                title='Latent space visualization - 2D PCA (After)', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        return temp_df
    
    @classmethod
    def analyze_DBSCAN(self, data: pd.DataFrame, repre: np.array, eps_list: list, min_samples_list: list, verbose: bool = True):
        print('--- ANALYSIS OF OUTLIERS REMOVAL USING DBSCAN ---') if verbose else None
        temp_df = data.copy()
        print('Data points: %d'%(temp_df.shape[0])) if verbose else None
        print('Representation shape: ',repre.shape) if verbose else None
        eps_list = list(set(eps_list))
        eps_list.sort()
        min_samples_list = list(set(min_samples_list))
        min_samples_list.sort()
        print('No. of Eps: %d   Start: %f   End: %f'%(len(eps_list), eps_list[0], eps_list[-1])) if verbose else None
        print('No. of Min samples: %d   Start: %d   End: %d'%(len(min_samples_list), min_samples_list[0], min_samples_list[-1])) if verbose else None
        
        result_df = pd.DataFrame(columns=['Eps', 'Min samples', 'No. of outliers', 'No. of data remaining'])
        
        config_list = [(eps, min_samples) for eps in eps_list for min_samples in min_samples_list]
        for eps, ms in tqdm(config_list, position=0, leave=True):
            temp_df = data.copy()
            outliers_idx = self.DBSCAN_to_outliers(repre, eps, ms)
            temp_df.drop(outliers_idx, inplace = True)
            temp_df.reset_index(drop=True, inplace = True)
            post_repre = np.delete(repre, outliers_idx, axis=0)
            result_df = pd.concat([result_df,pd.DataFrame({'Eps': [eps], 'Min samples': [ms], 'No. of outliers': [len(outliers_idx)], 'No. of data remaining': [temp_df.shape[0]]})], axis=0,ignore_index=True)
        
        result_df.reset_index(drop=True, inplace = True)
        return result_df
    
    @classmethod
    def Z_score_remove(self, data: pd.DataFrame, repre: np.array, activity_col: str = None, verbose: bool = True):
        print('--- REMOVE OUTLIERS USING Z-Score ---') if verbose else None
        temp_df = data.copy()
        print('Data points: %d'%(temp_df.shape[0])) if verbose else None
        print('Representation shape: ',repre.shape) if verbose else None
        
        print('** Original representation with 2D PCA **') if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values
        else:
            c = None
        pca = PCA_2D(repre)
        fig = pca.plot(repre, c=c, cmap='crest', hull=True,
                                title='Latent space visualization - 2D PCA (Before)', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        print('** Z-Score outlier removal **') if verbose else None
        z = np.abs(stats.zscore(repre))
        outliers_idx = np.where(z > 3)[0]
        print('Number of outliers removed: %d'%(len(outliers_idx))) if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values[outliers_idx]
        else:
            c = None
        fig2 = pca.plot(repre[outliers_idx], c=c, cmap='crest', hull=False,
                                title='Outliers - 2D PCA', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        temp_df.drop(outliers_idx, inplace = True)
        temp_df.reset_index(drop=True, inplace = True)
        post_repre = np.delete(repre, outliers_idx, axis=0)
        
        print('Final data points: %d\n'%(temp_df.shape[0])) if verbose else None
        
        print('** Post-re Representation with 2D PCA **') if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values
        else:
            c = None
        fig3 = pca.plot(post_repre, c=c, cmap='crest', hull=True,
                                title='Latent space visualization - 2D PCA (After)', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        return temp_df
    
    @classmethod
    def IQR_remove(self, data: pd.DataFrame, repre: np.array, activity_col: str = None, verbose: bool = True):
        print('--- REMOVE OUTLIERS USING IQR ---') if verbose else None
        temp_df = data.copy()
        print('Data points: %d'%(temp_df.shape[0])) if verbose else None
        print('Representation shape: ',repre.shape) if verbose else None
        
        print('** Original representation with 2D PCA **') if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values
        else:
            c = None
        pca = PCA_2D(repre)
        fig = pca.plot(repre, c=c, cmap='crest', hull=True,
                                title='Latent space visualization - 2D PCA (Before)', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        
        q1 = np.percentile(repre, 25, axis=0)
        q3 = np.percentile(repre, 75, axis=0)
        iqr = q3 - q1
        outliers_idx = np.where((repre < q1 - 1.5 * iqr) | (repre > q3 + 1.5 * iqr))[0]
        print('Number of outliers removed: %d'%(len(outliers_idx))) if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values[outliers_idx]
        else:
            c = None
        fig2 = pca.plot(repre[outliers_idx], c=c, cmap='crest', hull=False,
                                title='Outliers - 2D PCA', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        temp_df.drop(outliers_idx, inplace = True)
        temp_df.reset_index(drop=True, inplace = True)
        post_repre = np.delete(repre, outliers_idx, axis=0)
        
        print('Final data points: %d\n'%(temp_df.shape[0])) if verbose else None
        
        print('** Post-re Representation with 2D PCA **') if verbose else None
        if activity_col is not None:
            c = temp_df[activity_col].values
        else:
            c = None
        fig3 = pca.plot(post_repre, c=c, cmap='crest', hull=True,
                                title='Latent space visualization - 2D PCA (After)', 
                                title_fontsize = 18, title_x = 0.45, title_y = 1.0, verbose=False) if verbose else None
        plt.show() if verbose else None
        plt.close() if verbose else None
        return temp_df
    
def check_ref(df: pd.DataFrame, search_col: str, active_dict: dict, search_for: str = 'smiles' or 'name', active_col: str = None):
    active_df_dict = {}
    for key, value in active_dict.items():
        if search_for == 'smiles':
            active_df_dict[key] = df[df[search_col]==value].reset_index(drop=True)
        elif search_for == 'name':
            active_df_dict[key] = df[df[search_col].str.upper()==key.upper()].reset_index(drop=True)
    for key in active_dict.keys():
        if active_col is None:
            print('%s: %d' % (key, active_df_dict[key].shape[0]))
        else:
            print('%s: %d   mean pChEMBL Value: %.5f' % (key, active_df_dict[key].shape[0], active_df_dict[key][active_col].mean()))
    return active_df_dict

def change_qualitative_sign(activity_type, relating_value):
    if activity_type == 'IC50':
        if relating_value == '>':
            return '<'
        elif relating_value == '<':
            return '>'
        elif relating_value == '<=':
            return '>='
        elif relating_value == '>=':
            return '<='
        else:
            return relating_value
    if activity_type == 'pIC50':
        return relating_value

def handle_range_value(value, activity_type, keep='mean'):
    # keep = 'mean' | 'best' | 'worst'
    try:
        value = float(value)
    except:
        if activity_type == 'IC50':
            if keep == 'best':
                value = value.str.split(pat = '-', expand = True).min(axis =1)
            elif keep == 'worst':
                value = value.str.split(pat = '-', expand = True).max(axis =1)
            elif keep == 'mean':
                value = value.str.split(pat = '-', expand = True).mean(axis =1)
        if activity_type == 'pIC50':
            if keep == 'best':
                value = value.str.split(pat = '-', expand = True).max(axis =1)
            elif keep == 'worst':
                value = value.str.split(pat = '-', expand = True).min(axis =1)
            elif keep == 'mean':
                value = value.str.split(pat = '-', expand = True).mean(axis =1)
    return float(value)

def convert_activity(value, unit, activity_type, MW):
    if activity_type == 'IC50':
        if unit == 'μM':
            return -np.log10(value*1e-6)
        elif unit  == 'µM':
            return -np.log10(value*1e-6)
        elif unit  == 'ug.mL-1':
            return -np.log10(value*1e-6)
        elif unit  == 'nM':
            return -np.log10(value*1e-9)
        elif unit  == 'nmol/l':
            return -np.log10(value*1e-9)
        elif unit  == 'mM':
            return -np.log10(value*1e-3)
        elif unit  == 'M':
            return -np.log10(value*1)
        elif unit  == 'μg/ml':
            return -np.log10(value/MW*10e-3)
        elif unit  == 'µg/mL':
            return -np.log10(value/MW*10e-3)
        elif unit  == 'ng/mL':
            return -np.log10(value/MW*10e-6)
    if activity_type == 'pIC50':
        return value