import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer
import warnings
warnings.filterwarnings(action='ignore')
class rescale():
    """
    Rescale data to normal or range distribution

    Inputs:
    -------
    data_train: pandas.DataFrame
        Data for training model after cleaning
    data_test: pandas.DataFrame
        Data for external validation after cleaning
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    scaler_method: string ('MinMaxScaler', 'StandardScaler', 'RobustScaler')
        Scaling method

    Returns: Rescaled Data
    -------
    data_train: pandas.DataFrame
        Data train afer rescaling
    data_test: pandas.DataFrame
        Data test afer rescaling
    """
    def __init__(self, data_train, data_test, activity_col, smiles_col, scale_x = 'MinMaxScaler', scale_y = None, verbose=True):
        self.activity_col    = activity_col
        self.smiles_col = smiles_col
        self.scale_x = scale_x
        self.scale_y = scale_y
        
        self.data_train_0 = data_train
        self.data_test_0 = data_test
        
        self.Data_train = data_train.copy()
        self.Data_test = data_test.copy()
        self.verbose = verbose
        
        # self.scl1 = MinMaxScaler() 
        # self.scl2 = StandardScaler()  
        # self.scl3 = RobustScaler()   
        # self.scl4 = FunctionTransformer(lambda x: x)
    
    def fit(self):
        self.data_train = self.data_train_0.copy()
        self.data_test = self.data_test_0.copy()
        self.activity_col = self.activity_col
        self.data_train.reset_index(drop = True, inplace=True)
        self.data_test.reset_index(drop = True, inplace=True)
        df_train_int = self.data_train.drop([self.activity_col, self.smiles_col], axis = 1).select_dtypes("int64")
        # df_train_int = df_train_int.reset_index(drop = True)
        print("*"*75) if self.verbose else None
        if self.scale_y is not None:
            print("Scaling method for Y:", self.scale_y) if self.verbose else None
        if df_train_int.shape[1] == (self.data_train.shape[1]-2):
            if self.scale_y is not None:
                y_train = self.data_train[self.activity_col].to_numpy().reshape(-1,1)
                self.scale_y = globals()[self.scale_y]()
                self.scale_y.fit(y_train)
                # Y_train_trans = self.scale_y.transform(y_train)
                # df_y_train = pd.DataFrame(Y_train_trans, columns = [self.activity_col])
                # X_train = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("int64")
                # X_train.reset_index(drop = True, inplace = True)
                # self.data_train = pd.concat([df_y_train , X_train], axis = 1)
                
                # y_test = self.data_test[self.activity_col].to_numpy().reshape(-1,1)
                # Y_test_trans = self.scale_y.transform(y_test)
                # df_y_test = pd.DataFrame(Y_test_trans, columns = [self.activity_col])
                # X_test = self.data_test.drop([self.activity_col], axis = 1).select_dtypes("int64")
                # X_test.reset_index(drop = True, inplace = True)
                # self.data_test = pd.concat([df_y_test , X_test], axis = 1)
        else:
            # if self.scaler_method == 'MinMaxScaler':
            #     self.scl =self.scl1
            # elif self.scaler_method == 'StandardScaler':
            #     self.scl =self.scl2
            # elif self.scaler_method == 'RobustScaler':
            #     self.scl =self.scl3
            # else:
            #     self.scl =self.scl4
            
            

            #Train
            
            y_train = self.data_train[self.activity_col].to_numpy().reshape(-1,1)
            X_train = self.data_train.drop([self.activity_col, self.smiles_col], axis = 1).select_dtypes("float64").values

            print("Scaling method for X:", self.scale_x) if self.verbose else None
            self.scale_x = globals()[self.scale_x]()
            self.scale_x.fit(X_train)
            if self.scale_y is not None:
                self.scale_y = globals()[self.scale_y]()
                self.scale_y.fit(y_train)
            
            X_train_trans = self.scale_x.transform(X_train)
            # if self.scale_y is not None:
            #     Y_train_trans = self.scale_y.transform(y_train)
            # else:
            #     Y_train_trans = y_train
            idx = self.data_train.drop([self.activity_col, self.smiles_col], axis = 1).select_dtypes("float64").T.index

            df_X_train = pd.DataFrame(X_train_trans, columns = idx)
            # df_y_train = pd.DataFrame(Y_train_trans, columns = [self.activity_col])
            df_y_train = pd.DataFrame(y_train, columns = [self.activity_col])
            Data_train_float = pd.concat([df_y_train, df_X_train], axis = 1)

            self.data_train = pd.concat([self.data_train[[self.smiles_col]], Data_train_float , df_train_int], axis = 1)

                #test
            df_test_int = self.data_test.drop([self.activity_col, self.smiles_col], axis = 1).select_dtypes("int64")

            y_test = self.data_test[self.activity_col].to_numpy().reshape(-1,1)
            X_test = self.data_test.drop([self.activity_col, self.smiles_col], axis = 1).select_dtypes("float64").values


            X_test_trans = self.scale_x.transform(X_test)
            # if self.scale_y is not None:
            #     Y_test_trans = self.scale_y.transform(y_test)
            # else:
            #     Y_test_trans = y_test

            df_X_test = pd.DataFrame(X_test_trans, columns = idx)
            # df_y_test = pd.DataFrame(Y_test_trans, columns = [self.activity_col])
            df_y_test = pd.DataFrame(y_test, columns = [self.activity_col])
            Data_test_float = pd.concat([df_y_test, df_X_test], axis = 1)

            self.data_test = pd.concat([self.data_test[[self.smiles_col]],Data_test_float , df_test_int], axis = 1)
        
    def save_pipeline(self, SAVE_PREFIX):
        if self.scale_y is not None:
            save = SAVE_PREFIX+'/rescale_y.pkl'
            with open(save, 'wb') as f:
                pickle.dump(self.scale_y, f)
        if type(self.scale_x) != str:
            save = SAVE_PREFIX+'/rescale_x.pkl'
            with open(save, 'wb') as f:
                pickle.dump(self.scale_x, f)
