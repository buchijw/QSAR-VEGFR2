import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class Data_preprocess():
    """
    Preprocess data include:
     - Clean duplicated data (columns and rows)
     - Find percentage of missing value, choose suitable method to make imputation
     - Check variance threshold
     - Convert nomial to integer.
    
    Input:
    -----
    data_train: pandas.DataFrame
        Data for training model.
    data_test: pandas.DataFrame
        Data for external validation.
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    var_thresh: float
        Variance threshold to remove features

    Returns:
    --------
    Data_train and Data_test: pandas.DataFrame
        Data after preprocessing
    """
    def __init__(self, data_train, data_test, activity_col, smiles_col, var_thresh=0.05, impute_params = None, verbose=True):
        self.data_train= data_train.copy()
        self.data_test= data_test.copy()
        self.activity_col = activity_col
        self.var_thresh = var_thresh
        self.impute_params = impute_params if impute_params else {}
        self.verbose = verbose
        self.smiles_col = smiles_col
    
    # 1. Remove duplicate data - rows and columns
    def Duplicate_data(self):
        
        # 1.1. Duplicate rows: xóa hàng bị trùng 
        dup_rows = self.data_train.drop([self.smiles_col],axis=1).duplicated()
        print(f"Total duplicated rows-train: {(dup_rows == True).sum()}") if self.verbose else None
        print("Data train before drop duplicates:", self.data_train.shape[0]) if self.verbose else None
        feature = self.data_train.columns.tolist()
        feature.remove(self.smiles_col)
        self.data_train.drop_duplicates(inplace = True, subset=feature)
        self.data_train.reset_index(drop = True, inplace = True)
        print("Data train after drop duplicates:", self.data_train.shape[0]) if self.verbose else None
        print(75*"*") if self.verbose else None
        self.train_dup = (dup_rows == True).sum()
        
        # test
        dup_rows = self.data_test.drop([self.smiles_col],axis=1).duplicated()
        print(f"Total duplicated rows-test: {(dup_rows == True).sum()}") if self.verbose else None
        print("Data test before drop duplicates:", self.data_test.shape[0]) if self.verbose else None
        feature = self.data_train.columns.tolist()
        feature.remove(self.smiles_col)
        self.data_test.drop_duplicates(inplace = True, subset=feature)
        self.data_test.reset_index(drop = True, inplace = True)
        print("Data test after drop duplicates:", self.data_test.shape[0]) if self.verbose else None
        print(75*"*") if self.verbose else None
        self.test_dup = (dup_rows == True).sum()
       
        
        # 1.2. Duplicate columns: xóa cột bị trùng
          
        #matrix transpose
        dup = self.data_train.T[self.data_train.T.duplicated()]
        self.idx = dup.index
        print(self.idx) if self.verbose else None
        print(f"Total similar columns: {dup.shape[0]}") if self.verbose else None
        self.dup_col = dup.shape[0]
        #train
        print("Data train before drop duplicates:", self.data_train.shape) if self.verbose else None
        self.data_train.drop(self.idx, axis = 1, inplace = True) 
        print("Data after drop duplicates:", self.data_train.shape) if self.verbose else None
        print(75*"*") if self.verbose else None
        
        #test
        print("Data test before drop duplicates:", self.data_test.shape) if self.verbose else None
        self.data_test.drop(self.idx, axis = 1, inplace = True) 
        print("Data test after drop duplicates:", self.data_test.shape) if self.verbose else None
        print(75*"*") if self.verbose else None
        
    # 2. Check Variance Threshold: 
    def Variance_Threshold(self):
        y = self.data_train[self.activity_col]
        X = self.data_train.drop([self.activity_col, self.smiles_col], axis =1)
        print(X.shape, y.shape) if self.verbose else None
        while True:
            try:
               # Define thresholds to check
                thresholds = np.arange(0.0, 1, 0.05)
                # Apply transform with each threshold
                results = list()
                for t in thresholds:
                # define the transform
                    transform = VarianceThreshold(threshold=t)
                # transform the input data
                    X_sel = transform.fit_transform(X)
                # determine the number of input features
                    n_features = X_sel.shape[1]
                    print('>Threshold=%.2f, Features=%d' % (t, n_features)) if self.verbose else None 
                # store the result
                    results.append(n_features)
                break
            except:
                break
        # plot the threshold vs the number of selected features
        sns.set('notebook')
         # plot the threshold vs the number of selected features
        plt.figure(figsize=(14,8))
        plt.title("Variance analysis", fontsize = 24, weight = 'semibold')
        plt.xlabel("Variance threshold", fontsize = 16)
        plt.ylabel("Number of features", fontsize = 16)
       
        plt.plot(thresholds[:len(results)], results)
        #plt.savefig(self.SAVE_PREFIX +"var.png", dpi = 600)
        plt.show() if self.verbose else None
        plt.close()

    # 3. Remove variance Threshold:  
    def remove_low_variance(self, thresh):
        fea_df = self.data_train.drop([self.activity_col, self.smiles_col], axis=1)
        fea_df_test = self.data_test.drop([self.activity_col, self.smiles_col], axis=1)
        selector = VarianceThreshold(thresh)
        selector.fit(fea_df)
        features = selector.get_support(indices = False)
        # features[0]=True # target can not be removed
        self.data_train = pd.concat([self.data_train[[self.activity_col, self.smiles_col]],fea_df.loc[:, features]],axis=1)
        self.data_test = pd.concat([self.data_test[[self.activity_col, self.smiles_col]],fea_df_test.loc[:, features]],axis=1)
        print(75*"*") if self.verbose else None
        self.var = features.sum()
        self.var_cols = self.data_train.drop([self.activity_col, self.smiles_col], axis=1).columns
    

###########################################################################################
# MISSING VALUES
    # 4. Find Missing Percentage: 
    
    def find_missing_percent(self, data):
        """
        Returns dataframe containing the total missing values and percentage of total
        missing values of a column.
        """
        miss_df = pd.DataFrame({'ColumnName':[],'TotalMissingVals':[],'PercentMissing':[]})
        for col in data.columns:
            sum_miss_val = data[col].isnull().sum()
            percent_miss_val = round((sum_miss_val/data.shape[0])*100,2)
            missinginfo = {"ColumnName" : col, "TotalMissingVals" : sum_miss_val, "PercentMissing" : percent_miss_val}
            # miss_df = miss_df.append(missinginfo, ignore_index = True)
            miss_df = pd.concat([miss_df, pd.DataFrame([missinginfo])], ignore_index = True)

        miss_df = miss_df[miss_df["PercentMissing"] > 0.0]
        miss_df = miss_df.reset_index(drop = True)
        return miss_df
    
    # 5. Handle missing values
    def Missing_value_cleaning(self, impute = True, impute_method = "KNNImputer", KNN_neighbors = 5):
        # impute = True | False
        # impute_method = "mean" | "mode" | "median" | "KNNImputer" | "MICE_BR" | "MICE_RF" | "MICE_Grad"
        self.imp = None
        miss_df = self.find_missing_percent(self.data_train)
        print(miss_df) if self.verbose else None
        # Remove columns with high missing percentage
        miss_thresh= 50 # modify
        
        self.drop_cols = miss_df[miss_df['PercentMissing'] > miss_thresh].ColumnName.tolist()
        print("Drop_cols",  self.drop_cols) if self.verbose else None
        
            
        self.data_train.drop(self.drop_cols, axis =1, inplace = True)
        self.data_test.drop(self.drop_cols, axis =1, inplace = True)
        
        print("Total missing value-train", self.data_train.isnull().sum().sum()) if self.verbose else None
        print("Total missing value-test", self.data_test.isnull().sum().sum()) if self.verbose else None
        
        null_data_train = self.data_train[self.data_train.isnull().any(axis=1)]
        display(null_data_train) if self.verbose else None
        print("Total row-train with missing value", null_data_train.shape[0]) if self.verbose else None
        
        null_data_test = self.data_test[self.data_test.isnull().any(axis=1)]
        display(null_data_test) if self.verbose else None
        print("Total row-test with missing value", null_data_test.shape[0]) if self.verbose else None
        
        if null_data_train.shape[0] == 0:
            self.data_test = self.data_test.dropna(inplace = False)
            self.data_test.reset_index(drop = True, inplace = True)
        else:
            if not impute:
                Data_train = self.data_train.dropna(inplace = False)
                Data_train.reset_index(drop = True, inplace = True)

                Data_test = self.data_test.dropna(inplace = False)
                Data_test.reset_index(drop = True, inplace = True)
            if impute:
                if impute_method == 'mean':
                    self.imp=SimpleImputer(missing_values=np.NaN, strategy='mean')
                elif impute_method == 'median':
                    self.imp=SimpleImputer(missing_vlues=np.NaN, strategy='median')
                elif impute_method == 'mode':
                    self.imp=SimpleImputer(missing_values=np.NaN, strategy='mode')
                elif impute_method == 'KNNImputer':
                    self.imp=KNNImputer(n_neighbors=KNN_neighbors) 
                elif impute_method == 'MICE_BR':
                    estimator = BayesianRidge()
                    self.imp= IterativeImputer(random_state=42, missing_values=np.NaN, estimator= estimator)
                elif impute_method == 'MICE_Grad':
                    estimator = GradientBoostingRegressor(random_state = 42)
                    self.imp= IterativeImputer(random_state=42, missing_values=np.NaN, estimator= estimator)
                elif impute_method == 'MICE_RF':
                    estimator = RandomForestRegressor(random_state = 42)
                    self.imp= IterativeImputer(random_state=42, missing_values=np.NaN, estimator= estimator)
                
                self.imp.fit(self.data_train.drop([self.activity_col, self.smiles_col], axis=1))
                
                # train
                Data_train=pd.DataFrame(self.imp.transform(self.data_train.drop([self.activity_col, self.smiles_col], axis=1)))
                Data_train.columns=self.data_train.drop([self.activity_col, self.smiles_col], axis=1).columns
                Data_train.index=self.data_train.drop([self.activity_col, self.smiles_col], axis=1).index
                Data_train = pd.concat([self.data_train[[self.activity_col, self.smiles_col]], Data_train], axis=1)
                display(Data_train.shape) if self.verbose else None
                # test
                Data_test=pd.DataFrame(self.imp.transform(self.data_test.drop([self.activity_col, self.smiles_col], axis=1)))
                Data_test.columns=self.data_test.drop([self.activity_col, self.smiles_col], axis=1).columns
                Data_test.index=self.data_test.drop([self.activity_col, self.smiles_col], axis=1).index
                Data_test = pd.concat([self.data_test[[self.activity_col, self.smiles_col]], Data_test], axis=1)
                display(Data_test.shape) if self.verbose else None
                print(75*"*") if self.verbose else None
            self.data_train = Data_train
            self.data_test = Data_test
      
    
    # 6. Convert low unique columns to integer
 
    def Nomial(self):
        df = self.data_train.drop([self.smiles_col, self.activity_col],axis=1)
        data = df.loc[:, (df.nunique() <10).values & (df.max() <10).values]  #feature with unique < 10 and max value <10 will set to be int64
        self.nomial_cols = data.columns #select columns with int64
        #set all  col_olumn to int64
        #print("DINH TINH:", col)
        #savefile("Nomial_Col.txt", col)
        self.data_train[self.nomial_cols]=self.data_train[self.nomial_cols].astype('int64')  
        self.data_test[self.nomial_cols]=self.data_test[self.nomial_cols].astype('int64')    
        display(self.data_train.head(5)) if self.verbose else None
    
    
        
    # 7. Activate 
    def fit(self):
        self.Duplicate_data()
        self.Missing_value_cleaning(**self.impute_params) # missing values must be handled before feature selection
        self.Variance_Threshold()
        self.remove_low_variance(thresh= self.var_thresh)
        
        self.Nomial()

    def save_pipeline(self, SAVE_PREFIX):
        save = SAVE_PREFIX+'/DUP_COL.txt'
        with open(save, 'w') as f:
            f.write(str(list(self.idx)))
        save = SAVE_PREFIX+'/drop_cols.txt'
        with open(save, 'w') as f:
            f.write(str(list(self.drop_cols)))
        if self.imp is not None:
            save = SAVE_PREFIX+'/imp.pkl'
            with open(save, 'wb') as f:
                pickle.dump(self.imp, f)
        save = SAVE_PREFIX+'/Variance_Cols.txt'
        with open(save, 'w') as f:
            f.write(str(list(self.var_cols)))
        save = SAVE_PREFIX+'/Nomial_Cols.txt'
        with open(save, 'w') as f:
            f.write(str(list(self.nomial_cols)))
