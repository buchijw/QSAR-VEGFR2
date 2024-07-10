import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Scaffold_split import StratifiedScaffold_train_test_split, get_scaff


class Data_Integration():
    """
    Create Data Frame from csv file, find missing value (NaN), choose a threshold to make target transformation (Classification)
    remove handcrafted value (Regression), split Data to Train and Test and show the chart
    
    Input:
    -----
    data : pandas.DataFrame
        Data with features and target columns
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
   
        
    Returns:
    --------
    Data_train: pandas.DataFrame
        Data for training model
    Data_test: pandas.DataFrame
        Data for external validation  
    """
    def __init__(self, data, activity_col, smiles_col, task_type, SAVE_PREFIX='./', target_thresh = None, verbose=True):
        
        self.data = data
        self.activity_col= activity_col
        self.task_type = task_type
        if self.task_type.title() == "C":
            assert target_thresh is not None
        self.target_thresh = target_thresh
        self.verbose=verbose
        
        self.smiles_col = smiles_col
        
        self.SAVE_PREFIX = SAVE_PREFIX
        
    # 1. Check nan value - Mark Nan value to np.nan
    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            try:
                float(value)
                if np.isnan(value):
                    index.append(key)
                else:
                    continue
            except:
                index.append(key)
        if len(index) != 0:
            data[index] = np.nan 
    
    # 2. Target transformation - Classification
    def target_bin(self, thresh, input_target_style = 'pIC50'):
        if input_target_style != 'pIC50':
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 1
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 0
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        else:
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 0
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 1
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        
    
    # 3. Split data
    #Chia tập dữ liệu  thành tập train và test.
    def Data_split(self):
        
        if self.task_type.title() == "C":
            if len(self.data[self.activity_col].unique()) ==2: 
                y = self.data[self.activity_col]
            else:
                self.target_bin(thresh = self.target_thresh)
                y = self.data[self.activity_col]
            
            self.stratify = y
        
        elif self.task_type.title() == "R":
            y = self.data[self.activity_col]
            self.stratify = None
            
        
       
        X = self.data.drop([self.activity_col, self.smiles_col], axis =1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, 
        #                                                     random_state = 42, stratify = self.stratify
        data_train, data_test = StratifiedScaffold_train_test_split(self.data,self.smiles_col,self.activity_col,n_splits = 7, random_state=42, shuffle = True, scaff_based = 'median')
        
        if self.verbose:
            print("Train: {:.2f}%".format(len(data_train)/len(self.data)*100))
            print("Test: {:.2f}%".format(len(data_test)/len(self.data)*100))
            print("Train/Test: {:.2f}".format(len(data_train)/len(data_test)))

            tr_scaff = set(get_scaff(data_train[self.smiles_col]).keys())
            ts_scaff = set(get_scaff(data_test[self.smiles_col]).keys())
            print(f"Number of overlapping scaffolds: {len(tr_scaff.intersection(ts_scaff))}")
        
        X_train = data_train.drop([self.activity_col, self.smiles_col], axis =1)
        y_train = data_train[[self.activity_col]]
        X_test = data_test.drop([self.activity_col, self.smiles_col], axis =1)
        y_test = data_test[[self.activity_col]]


        #index:
        self.idx = X.T.index

        #Train:
        self.df_X_train = pd.DataFrame(X_train, columns = self.idx)
        self.df_y_train = pd.DataFrame(y_train, columns = [self.activity_col])
        self.data_train = pd.concat([self.df_y_train, data_train[[self.smiles_col]], self.df_X_train], axis = 1)
        

        #test
        self.df_X_test = pd.DataFrame(X_test, columns = self.idx)
        self.df_y_test = pd.DataFrame(y_test, columns = [self.activity_col])
        self.data_test = pd.concat([self.df_y_test, data_test[[self.smiles_col]], self.df_X_test], axis = 1)
        
        print("Data train:", self.data_train.shape) if self.verbose else None
        print("Data test:", self.data_test.shape) if self.verbose else None
        print(75*"*") if self.verbose else None
        

    def Visualize_target(self):
        if self.task_type.title() == "C":
            sns.set('notebook')
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            plt.title(f'Training data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_train[self.activity_col])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_train[self.activity_col].values == 1).sum() / (self.data_train[self.activity_col].values == 0).sum(),3))}')
            plt.subplot(1,2,2)
            plt.title(f'External data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_test[self.activity_col])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_test[self.activity_col].values == 1).sum() / (self.data_test[self.activity_col].values == 0).sum(),3))}')
            plt.savefig(f"{self.SAVE_PREFIX}distribution.png", dpi = 600, bbox_inches="tight")
            plt.savefig(f"{self.SAVE_PREFIX}distribution.pdf", dpi = 600, format='pdf',transparent = False,bbox_inches="tight")
            plt.savefig(f"{self.SAVE_PREFIX}distribution.svg", dpi = 600, format='svg',transparent = False,bbox_inches="tight")
            plt.savefig(f"{self.SAVE_PREFIX}distribution.eps", dpi = 600, format='eps',transparent = False,bbox_inches="tight")
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show() if self.verbose else None
        else:
            sns.set('notebook')
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            sns.histplot(self.data_train[self.activity_col], palette = 'deep', kde = True)
            #plt.hist(self.Data_train.iloc[:,0])
            plt.title(f'Train set distribution', weight = 'semibold', fontsize = 16)
            plt.subplot(1,2,2)
            #plt.hist(self.Data_test.iloc[:,0])
            sns.histplot(self.data_test[self.activity_col], palette = 'deep', kde = True)
            plt.title(f'External validation set distribution',weight = 'semibold', fontsize = 16)
            plt.savefig(f"{self.SAVE_PREFIX}distribution.png", dpi = 600, bbox_inches="tight")
            plt.savefig(f"{self.SAVE_PREFIX}distribution.pdf", dpi = 600, format='pdf',transparent = False,bbox_inches="tight")
            plt.savefig(f"{self.SAVE_PREFIX}distribution.svg", dpi = 600, format='svg',transparent = False,bbox_inches="tight")
            plt.savefig(f"{self.SAVE_PREFIX}distribution.eps", dpi = 600, format='eps',transparent = False,bbox_inches="tight")
            # plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show() if self.verbose else None
        plt.close()
         
    def fit(self):
        df = self.data.drop([self.smiles_col],axis=1)
        df.apply(self.Check_NaN)
        self.data = pd.concat([self.data[[self.smiles_col]], df], axis=1)
        self.Data_split()
        self.Visualize_target()