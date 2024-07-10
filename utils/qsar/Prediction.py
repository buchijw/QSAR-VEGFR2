import pickle
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, KBinsDiscretizer
from sklearn.compose import TransformedTargetRegressor

class predict():
    def __init__(self, materials_path ,data, ID, impute = False):
        self.SAVE_PREFIX = materials_path

        self.data = data
        self.data_pre = self.data.copy()
        self.data_pre.columns = [str(i) for i in self.data_pre.columns]
        self.ID = ID
        self.impute = impute
        

    def pickleload(self,fname):
        with open(self.SAVE_PREFIX + fname,'rb') as f:
            return pickle.load(f)

    def loadfile(self,fname):
        with open(self.SAVE_PREFIX + fname,'r') as f:
            return eval(f.read())


    def prepare_data_pred(self):
        self.data_pred = self.data.drop(self.ID,axis =1)
        #self.data_pre = self.data_pre


    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan

    def dupcol(self):
        DUP = self.loadfile("DUP_COL.txt")
        self.data_pre.drop(DUP, axis = 1, inplace = True)
        self.data_pre.reset_index(drop = True, inplace = True)

    def Missing_value_cleaning(self):
        drop_cols = self.loadfile("drop_cols.txt")
        self.data_pre.drop(drop_cols, axis =1, inplace = True)
        
        if self.impute:
            # impute
            imp = self.pickleload("imp.pkl")
            Data_train=pd.DataFrame(imp.transform(self.data_pre))
            Data_train.columns=self.data_pre.columns
            Data_train.index=self.data_pre.index
            self.data_pre = Data_train

    def variance_threshold(self):
        # remove target column
        FEATURES = self.loadfile("Variance_Cols.txt")
        self.data_pre = self.data_pre.loc[:, FEATURES]

    def Nomial(self):
        DINHTINH = self.loadfile("Nomial_Cols.txt")
        self.data_pre[DINHTINH]=self.data_pre[DINHTINH].astype('int64')





    def features_selection(self):
        self.X_pre = self.data_pre
        #Load model
        self.select = self.pickleload('feature_select.pkl')
        self.X_pre = self.select.transform(self.X_pre)
        return self.X_pre

    def model_predict(self):
        self.num_molecules = self.data_pre.shape[0]
        self.model = self.pickleload("model.pkl")
        
        ## for not TransformedTargetRegressor
        # self.y_pre_raw = self.model.predict(self.X_pre)
        # if self.scale_y is not None:
        #     self.y_pre = self.scale_y.inverse_transform(self.y_pre_raw.reshape(-1,1)).reshape(-1)
        # else:
        #     self.y_pre = self.y_pre_raw
        # self.Report={'ID': self.data[self.ID].values,
        # "Predict": self.y_pre,
        # "Predict_raw": self.y_pre_raw}
        
        # for TransformedTargetRegressor
        self.y_pre = self.model.predict(self.X_pre)
        self.Report={'ID': self.data[self.ID].values,
        "Predict": self.y_pre}

        self.report = pd.DataFrame(self.Report)
        return self.report
        
    def Rescale(self):
        df_int = self.data_pre.select_dtypes("int64")
        df_int = df_int.reset_index(drop = True)
        self.scale_y = None
        self.scale_x = None
        if os.path.exists(self.SAVE_PREFIX + "/rescale_y.pkl"):
            self.scale_y = self.pickleload('rescale_y.pkl')
        if os.path.exists(self.SAVE_PREFIX + "/rescale_x.pkl"):
            self.scale_x = self.pickleload('rescale_x.pkl')
            # if df_int.shape[1] != (self.data_pre.shape[1]-1):
            x_float = self.data_pre.select_dtypes("float64").values
            x_float_trans = self.scale_x.transform(x_float)
            idx_float = self.drar_pre.select_dtypes("float64").T.index
            df_float = pd.DataFrame(x_float_trans, columns = idx_float)
            self.data_pre = pd.concat([df_float , df_int], axis = 1)


    def predict(self, verbose = True):
        print('--- PREDICTION ---') if verbose else None
        print('Preparing data...') if verbose else None
        print(' ⊢ Drop ID...') if verbose else None
        self.prepare_data_pred()
        print(' ⊢ Check NaN...') if verbose else None
        self.data_pre.apply(self.Check_NaN)
        print(' ⊢ Drop duplicated columns...') if verbose else None
        self.dupcol()
        print(' ⊢ Missing value cleaning...') if verbose else None
        self.Missing_value_cleaning()
        print(' ⊢ Applying variance_threshold...') if verbose else None
        self.variance_threshold()
        print(' ⊢ Nomial columns...') if verbose else None
        self.Nomial()
        print(' ⊢ Rescaling...') if verbose else None
        self.Rescale()
        print('Feature selection...') if verbose else None
        self.features_selection()
        print('Predicting...') if verbose else None
        self.model_predict()
        print('Done!') if verbose else None
        return self.report