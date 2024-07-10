import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from Data_integration import Data_Integration
from Data_preprocess import Data_preprocess
from Rescale import rescale
from Feature_selection import feature_selection_pipeline
from Model_selection import model_selection
from Meta_analysis import statistic_data, statistic_test
import warnings
warnings.filterwarnings('ignore')
sns.set('notebook')

class QSAR_pipeline:
    """
    - QSAR data pipeline includes:
        - Data integration
        - Data preprocessing
        - Outliers handling
        - Rescaling
        - Feature selection
        - Model building
        - Statistic analysis

    Input:
    ------
    data_path: path
        Directory contains all data types to run QSAR pipeline
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    var_thresh: float (default = 0.05)
         Variance threshold to remove features
    save_data: bool (default = False)
        True if want to save the posthoc analysis data
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    refinement: bool (default = False)
        True to remove results equal to zero in boxplot visualization
    
    posthoc_method: list of str (['Wilcoxon'], ['Mannwhitney'], ['Wilcoxon', 'Mannwhitney']) 
        Method for posthoc analysis
    kind_analysis: list of str (['Meta', 'Subgroup']):
        Kind of posthoc analysis
    
    Returns:
    --------
    Data_posthoc: new directory (data_dir in posthoc test)
        Directory contains data for posthoc analysis
    Meta: new directory
        Directory contains posthoc data and figures
    

    --------
    
    """
    def __init__(self, data_path, activity_col, smiles_col, kind_analysis=['Meta'],posthoc_method =['Wilcoxon'], var_thresh = 0.05,
                 task_type  ='C', scoring ='f1',refinement = False, save_data = False, target_thresh=7, 
                 scale_x = 'MinMaxScaler', scale_y = 'StandardScaler', verbose=True):
       
        self.data_path = data_path
        self.activity_col = activity_col
        self.kind_analysis = kind_analysis
        self.task_type = task_type
        self.scoring = scoring
        self.posthoc_method = posthoc_method
        self.refinement = refinement
        self.save_data = save_data # saving meta data
        self.target_thresh = target_thresh
        self.var_thresh = var_thresh
        self.scale_x = scale_x
        self.scale_y = scale_y
        
        self.meta_folder = "/Data_posthoc"
        self.data_save_dir = f"{self.data_path}/{self.meta_folder}/"
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.data_save_dir )
        if not isExist:
           # Create a new directory because it does not exist
            os.makedirs(self.data_save_dir)
            #print("The new directory is created!")
        
        # Data raw tracking
        self.feature_0     = []
        self.dup_train     = []
        self.dup_test      = []
        self.dup_col       = []
        self.variance      = []
        self.Outlier_train = []
        self.Outlier_test  = []
        self.feature       = []
        
        self.training_data = []
        self.testing_data = []
        
        self.index = ['Feature', 'Duplicated data train', 'Duplicated data test','Duplicated columns','Variance', 
                      'Outlier_train', "Outlier_test", 'Feature_training', 'Training data','Testing data']
        
        self.smiles_col = smiles_col
        
        self.verbose = verbose
        
        
    def Data_pipeline(self, data, data_name):
        # 1. Data integration
        integration = Data_Integration(data=data, activity_col=self.activity_col, smiles_col=self.smiles_col, SAVE_PREFIX=f"{self.data_save_dir}/{data_name}",
                                       task_type =self.task_type, target_thresh=self.target_thresh, verbose=self.verbose)
        integration.fit()
        Data_train = integration.data_train
        Data_test = integration.data_test
        self.feature_0.append(Data_train.shape[1]-1)
        del integration
              

        # 2. Data preprocess
        preprocess = Data_preprocess(Data_train, Data_test, var_thresh = self.var_thresh, smiles_col=self.smiles_col,
                                     activity_col =self.activity_col, verbose=self.verbose)
        preprocess.fit()
        self.variance.append(preprocess.var)
        self.dup_train.append(preprocess.train_dup)
        self.dup_test.append(preprocess.test_dup) 
        self.dup_col.append(preprocess.dup_col) 
        
        Data_train = preprocess.data_train
        Data_test = preprocess.data_test
        del preprocess

        # 3. Rescale

        rescaling = rescale(Data_train,Data_test, smiles_col=self.smiles_col,
                            activity_col=self.activity_col, scale_x= self.scale_x, scale_y= self.scale_y, verbose=self.verbose)
        rescaling.fit()
        Data_train = rescaling.data_train
        Data_test = rescaling.data_test
        scale_y = rescaling.scale_y
        display(Data_train.head(3))  if self.verbose else None
        del rescaling
        
        # 4. Feature selection
        feature = feature_selection_pipeline(data_train=Data_train, data_test=Data_test, smiles_col=self.smiles_col,
                                             activity_col=self.activity_col,task_type =self.task_type,
                                             scoring = self.scoring, method ='RF', scale_y=scale_y)
        feature.fit()
        X_train = feature.X_train_new
        X_test  = feature.X_test_new
        y_train = feature.y_train
        y_test  = feature.y_test
        smiles_train = feature.smiles_train
        smiles_test = feature.smiles_test
        del feature
        del Data_train
        del Data_test
        
        self.feature.append(X_train.shape[1])
        self.training_data.append(X_train.shape[0])
        self.testing_data.append(X_test.shape[0])

        # 5. Model building
        model = model_selection(X_train, y_train, SAVE_PREFIX=self.data_save_dir, smiles_list=smiles_train,
                           data_name=data_name, task_type =self.task_type,scoring = self.scoring, scale_y=scale_y)
        model.compare()
        del model
        
    def fit(self):
        self.data_name = []
        for i in sorted(glob.glob(self.data_path+"/*.csv")):
            self.data_name.append(i[len(self.data_path)+1:-4])
        for i in self.data_name:
            data_path = str(i)+'.csv'
            print(f"Data name: {data_path}")
            data = pd.read_csv(f"{self.data_path}/{data_path}")
            self.Data_pipeline(data=data, data_name = i)
            del data
            
        for i in self.posthoc_method:
            for j in self.kind_analysis:
                static = statistic_data(data_dir=self.data_save_dir, save_data=self.save_data, refinement =self.refinement,  
                                    scoring = self.scoring, posthoc_method =i, 
                                    kind_analysis=j)
                static.fit()
            

       
        
        self.df_process = pd.DataFrame([self.feature_0, self.dup_train, self.dup_test, self.dup_col,
                                        self.variance, self.Outlier_train, self.Outlier_test,
                                       self.feature, self.training_data, self.testing_data], index = self.index).T
        # self.df_process = pd.DataFrame([self.feature_0,
        #                                 self.variance, self.Outlier_train, self.Outlier_test,
        #                                self.feature, self.training_data, self.testing_data], index = self.index).T
        self.df_process.index = self.data_name
        self.df_process.to_csv(f"{self.data_save_dir}/Meta_folder/Processing.csv")
        
