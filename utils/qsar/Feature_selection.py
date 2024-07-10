# Classification
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, chi2, RFE, RFECV, f_classif,mutual_info_regression,f_regression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import LinearSVC, SVC

# Regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.svm import LinearSVR, SVR
from sklearn.compose import TransformedTargetRegressor

import warnings
warnings.filterwarnings('ignore')

class FeatureTransformedTargetRegressor(TransformedTargetRegressor):
    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_

    @property
    def coef_(self):
        return self.regressor_.coef_

class feature_selection_pipeline:
    """
    - Remove unnecessary features
    - Show a chart that compares the effectiveness of each method.
    - Based on the chart, choose the best method.

    Input:
    ------
    data_train: pandas.DataFrame
        Data for training model after rescaling
    data_test: pandas.DataFrame
        Data for external validation after rescaling
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str
        Classification (C) or Regression (R)
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    method: str (default = 'RandomForest')
        Feature selection method include: 'Statistic_Anova','Statistic_Mutual','RandomForest',
        'ExtraTree', 'AdaBoost', 'GradBoost', 'XGBoost', 'Linear', 'SVM', 'RFE'
    
    Returns:
    --------
    Completed Data_train and Data_test
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    """
    def __init__(self, data_train, data_test, activity_col, smiles_col, task_type ='C',scoring = 'f1', method ='RF',
                scale_y = None, verbose=True, umap_obj=None):
        
        self.activity_col = activity_col
        self.task_type = task_type
        self.scoring = scoring
        self.method = method
        
        self.scale_y = scale_y
        self.verbose = verbose
        
        self.smiles_col = smiles_col
        self.smiles_train = np.asarray(data_train[smiles_col].tolist(),dtype=str)
        self.smiles_test = np.asarray(data_test[smiles_col].tolist(),dtype=str)
        self.umap_obj = umap_obj
        
        if self.scale_y is not None:
            print('Using Transformed Target Regressor') if self.verbose else None
        
        self.X_train = data_train.drop([self.activity_col,self.smiles_col], axis = 1)
        self.y_train = data_train[self.activity_col]
        self.X_test = data_test.drop([self.activity_col,self.smiles_col], axis = 1)
        self.y_test = data_test[self.activity_col]
        # Check task type
        if len(self.y_train.unique())==2:
            self.task_type == 'C'
        else:
            self.task_type == 'R'
    
    # 1. Statistic
    def Statistic(self):
        if self.task_type =='C':
            if self.method =='Anova':
                self.select = SelectKBest(score_func= f_classif, k= 50)
            elif self.method =='Mutual information':
                self.select = SelectKBest(score_func= lambda X,y: mutual_info_classif(X,y,random_state=42), k= 50)
        else:
            if self.method =='Anova':
                self.select = SelectKBest(score_func= f_regression, k= 50)
            elif self.method =='Mutual information':
                self.select = SelectKBest(score_func= lambda X,y: mutual_info_regression(X,y,random_state=42), k= 50)
        if self.scale_y is not None:
            self.select.fit(self.X_train, self.scale_y.transform(self.y_train.to_numpy().reshape(-1,1)))
        else:
            self.select.fit(self.X_train, self.y_train)
    
    # 2. Random forest
    def random_forest(self):
        if self.task_type =='C':
            forest = RandomForestClassifier(random_state=42)
        else:
            forest = RandomForestRegressor(random_state=42)
            if self.scale_y is not None:
                forest = FeatureTransformedTargetRegressor(regressor=forest, transformer=self.scale_y)
        forest.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(forest, prefit=True)
        
    
    # 3. ExtraTree
    def extra_tree(self):
        if self.task_type =='C':
            ext_tree = ExtraTreesClassifier(random_state=42)
        else:
            ext_tree = ExtraTreesRegressor(random_state=42)
            if self.scale_y is not None:
                ext_tree = FeatureTransformedTargetRegressor(regressor=ext_tree, transformer=self.scale_y)
        ext_tree.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(ext_tree, prefit=True)
    
    # 4. Ada
    def ada(self):
        if self.task_type =='C':
            ada = AdaBoostClassifier(random_state=42)
        else:
            ada = AdaBoostRegressor(random_state=42)
            if self.scale_y is not None:
                ada = FeatureTransformedTargetRegressor(regressor=ada, transformer=self.scale_y)
        ada.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(ada, prefit=True)
        
    # 5. Gradient boosting
    def grad(self):
        if self.task_type =='C':
            grad = GradientBoostingClassifier(random_state=42)
        else:
            grad = GradientBoostingRegressor(random_state=42)
            if self.scale_y is not None:
                grad = FeatureTransformedTargetRegressor(regressor=grad, transformer=self.scale_y)
        grad.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(grad, prefit=True)
    
    # 6. XGBoost
    def XGb(self):
        if self.task_type =='C':
            XGb = XGBClassifier(random_state = 42, verbosity=0, eval_metrics ='logloss')
        else:
            XGb = XGBRegressor(random_state = 42, verbosity=0,  eval_metrics ='logloss')
            if self.scale_y is not None:
                XGb = FeatureTransformedTargetRegressor(regressor=XGb, transformer=self.scale_y)
        XGb.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(XGb, prefit=True)
        
    # 7. Linear method
    def Linear(self):
        if self.task_type =='C':
            lasso = LogisticRegression(random_state=42, penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, max_iter = 10000)
        else:
            lasso = LassoCV(random_state = 42)
            if self.scale_y is not None:
                lasso = FeatureTransformedTargetRegressor(regressor=lasso, transformer=self.scale_y)
        lasso.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(lasso, prefit=True)
        
  
        
    def fit(self):
        if self.method == 'Anova':
            self.Statistic()
        elif self.method == 'Mutual information':
            self.Statistic()
        elif self.method =='RF':
            self.random_forest()
        elif self.method =='ExT':
            self.extra_tree()
        elif self.method =='Ada':
            self.ada()
        elif self.method =='Grad':
            self.grad()
        elif self.method =='XGB':
            self.XGb()
        elif self.method =='Linear':
            self.Linear()
        
        if self.method !='RFE':
            self.X_train_new = self.select.transform(self.X_train)
            self.X_test_new = self.select.transform(self.X_test)
        
    def save_pipeline(self, SAVE_PREFIX):
        save = SAVE_PREFIX+'/feature_select.pkl'
        with open(save, 'wb') as f:
            pickle.dump(self.select, f)
        
        save = SAVE_PREFIX+'/X_train.npy'
        with open(save, 'wb') as f:
            if self.method !='RFE':
                np.save(f, self.X_train_new)
            else:
                np.save(f, self.X_train)
        
        save = SAVE_PREFIX+'/X_test.npy'
        with open(save, 'wb') as f:
            if self.method !='RFE':
                np.save(f, self.X_test_new)
            else:
                np.save(f, self.X_test)
        
        self.y_train.to_csv(SAVE_PREFIX+'/y_train.csv', index=False)
        self.y_test.to_csv(SAVE_PREFIX+'/y_test.csv', index=False)
        
        save = SAVE_PREFIX+'/smiles_train.npy'
        with open(save, 'wb') as f:
            np.save(f, self.smiles_train)
        
        save = SAVE_PREFIX+'/smiles_test.npy'
        with open(save, 'wb') as f:
            np.save(f, self.smiles_test)
