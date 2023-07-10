# -*- coding: utf-8 -*-
# # Package setting

import pandas as pd


# # - Configure setting

# +
# - Code variable settings object
class Configure():
    def __init__(self, pathdata):
        
        # - Initial Settings (description) =====================================================
        # -- data path set 
        path_file = "__sample__/iris_regression_2.csv"
        self.df_path = pathdata + path_file
        print("Feature select (data path set done)===================")
        
        # -- cols for train ( read excel )
        #    (Note) get has tagged feature
        self.cols_train = ["ID","Sepal_Length","Sepal_Width","Petal_Length","Petal_Width","Label_num"]
        print("Feature select (cols_train read) done ==========")
        
        
        # - Feature Engineering=================================================================
        # -- columns rename 
        self.cols_rename = {"Sepal_Length" : "Sepal_Length_v2"}
        print("Feature Engineering (cols yn to 01) done==========")
        # -- columns ny to 01
        self.cols_nyto01 = []
        
        # -- column set ID 
        self.col_id = "ID"
        # -- column set Y 
        self.col_y = "Label_num"
        # -- columns need to onehot
        self.cols_onehot = []
        
        
        # - Model Training=================================================================
        self.model_set = {
            "model_name" : "iris_xgboost_lgbm",
            "algo" : "LGBM",
            "metric" : "rmse"
        }
        self.params = {
            'objective':'regression',
            'metric':'none',
            'n_jobs' : 1,
            'learning_rate' : 0.1,
            'verbose' : 1,
            'random_state' : 0,
            'n_estimators' : 7500
        }
        self.columns_valid_keep = ["ID", "Sepal_Length", "Label_num"]
        self.columns_drop = ["ID", "Label_num"]
        
        
        print("Model Training setting done ==============")
    def _get_pairs_columns (self, df, name_key, name_value, null_ignore = True) :
        """
        Get pairs coluns from dataframe, setting "name_key" and "name_value"
        to Dictionary 
        
        Args:
            df (pandas.DataFrame) : target dataframe for creating dictionary
            name_key (string) : key column name
            name_value (string) : value column name
            null_ignore (boolean) : if name_value null will ignore the dictionary set  
        
        Return:
            dict : all value mapping dictionary (ex: {name_key_value1 : name_value_value1, 
                                                  name_key_value2 : name_value_value2,.... })
        
        Note : 
        
        """
        if null_ignore : 
            df_cleaned = df.loc[~df[name_value].isna(),]
        pairs_columns = dict(zip(df_cleaned[ name_key ], df_cleaned[ name_value ]))
        
        print("column <{0}> and column <{1}> pairs done".format(name_key, name_value))
        
        return pairs_columns
        
        
        
