#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
#
# Docstring and Package Setting 

# %%
"""
Template Version : 1.0.0
    Released Date : 20201102
    Packages (version is reference only): 
        configure.py
        main.func.data_dsc
        main.func.feature_engn
        main.func.model_trn
        main.func.__load_modules__
        mlflow==1.11.0
        sweetviz==1.0beta6 
        pandas==1.1.2
        sklearn==0.21.2
        joblib==0.13.2 # main.func.model_trn
        numpy==1.16.3 # main.func.model_trn
        xgboost==0.90 # main.func.model_trn
        lightgbm==2.2.3 # main.func.model_trn
        

# ============== Project Define ==========================

Name : iris binary target train model
Version : 1.0.0
Note : 
    target : Is versicolor iris? (Y/N)
        - only test
Packages: 
    configure.py
    main.func.data_dsc
    main.func.feature_engn
    main.func.model_trn
    main.func.__load_modules__
    mlflow==1.11.0
    sweetviz==1.0beta6 
    pandas==1.1.2
    sklearn==0.21.2
    joblib==0.13.2 # main.func.model_trn
    numpy==1.16.3 # main.func.model_trn
    xgboost==0.90 # main.func.model_trn
    lightgbm==2.2.3 # main.func.model_trn
"""

# %% [markdown]
# ##################################################################
# ## - System packages and path setting
# ##################################################################

# %%
# - System Package import
import os
import sys
from imp import reload

# -- ROOT paths setting (Default to project root dir)
ROOT = 'E:/python3/_FINAL/Projects/Code_sample//'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
print(os.getcwd())
print(sys.path)

# %% [markdown]
# ##################################################################
# ##  - Packages, Settings, Reload 
# ##################################################################

# %%
# -- Packages Reloading ( Custom )  
from main.func import *
import configure
reload(configure)
__load_modules__.reload_modules(path_func = "main/func/", var_dict = locals())

# -- Package Setting ( Custom )
from main.func import data_dsc
from main.func import feature_engn
from main.func import model_trn
from main.func import __load_modules__

# -- Packages import ( Installed )
import mlflow
import sweetviz as sv       # 1.1 can use param <open_browser>
import pandas as pd 
from sklearn import metrics # Mlflow

# -- Basic paths Setting
PATH_DATA = 'data/'
PATH_OUTPUT = 'main/__sample__/output/'
PATH_TRACK = 'proj_track/'
# -- jupyternotebook Setting (tab the package too slow (avoid memory exhausted))
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')d


# -- Config Setting
CONF = configure.Configure(pathdata=PATH_DATA)

# %% [markdown]
# # Main ==========================================================

# %% [markdown]
# ##################################################################
# ## - Initial Setting
# ##################################################################

# %%
# -- Read Data
df_train = pd.read_csv(CONF.df_path, low_memory=False)
print("read_csv done ==== ")

# %% [markdown]
# ##################################################################
# ## - Data Description (for configure)
# ##################################################################

# %%


# -- Train Y Setting (NY to 01)
coly_o = df_train[CONF.col_y].copy()
df_train[CONF.col_y] = df_train[CONF.col_y].replace({"Y": "1",
                                                     "N": "0"})
df_train[CONF.col_y] = df_train[CONF.col_y].astype("int8")

# -- Data descripton save ( by PATH_OUTPUT )
df_descr = data_dsc.explore_df(df=df_train)
df_descr.to_csv(PATH_OUTPUT + "conf_feature_describe.csv")

# -- Sweet viz save ( by PATH_OUTPUT )
feature_config = sv.FeatureConfig(skip=[],                                      # 要忽略哪個特徵
                                  force_cat=[],                                 # Categorical特徵
                                  force_num=[],                                 # Numerical特徵
                                  force_text=None
                                  )                                             # Text特徵
report_train_with_target = sv.analyze([df_train, 'train'],
                                      target_feat=CONF.col_y,                   # 加入特徵變數
                                      feat_cfg=feature_config,
                                      pairwise_analysis="off"                   # pairwise 2million, 200 column 預估要 37小時
                                      )
report_train = sv.analyze([df_train, 'train'],
                          feat_cfg=feature_config,
                          pairwise_analysis="off"                               # pairwise 2million, 200 column 預估要 37小時
                          )

report_train_with_target.show_html( filepath=PATH_OUTPUT + 'Sweetviz_report_with_target.html')
report_train.show_html(filepath=PATH_OUTPUT + 'Sweetviz_report_columns.html')

# -- Train Y Setting (01 to NY)
df_train[CONF.col_y] = coly_o
print("====Base explotary done, path is <{0}>====".format(PATH_OUTPUT))




# %% [markdown]
# ##################################################################
# ## - Feature Engineering 
# ##################################################################

# %%


#  -- [CONF.cols_rename] (rename) 
df_train = df_train.rename(CONF.cols_rename)
#  -- [CONF.cols_nyto01] (replace values)
df_train = feature_engn.replace_values(df_train, 
                                       columns = list(set(CONF.cols_train).intersection(set(CONF.cols_nyto01))), 
                                       rule_replace = {"Y":"1", "N":"0", "NON":"-1", "" : "-2"}, 
                                       type_set = "int8",
                                       missing_replace = "-2")
# -- [CONF.cols_train] (get columns SELECT)
df_train = df_train[CONF.cols_train]

# -- [Sepal_Length_SQ] (created by Sepal_Length)
df_train["Sepal_Length_SQ"] = df_train["Sepal_Length"]**2

# -- [LengthxWidth] (created by Sepal_Length and Sepal_Width)
df_train["LengthxWidth"] = df_train["Sepal_Length"]*df_train["Sepal_Width"]




# %% [markdown]
# ##################################################################
# ## - Data Description (before training)
# ##################################################################

# %%


# -- Data descripton save ( by PATH_OUTPUT )
df_descr_beftrain = data_dsc.explore_df(df = df_train)
df_descr_beftrain.to_csv(PATH_OUTPUT + "conf_feature_describe(before_train).csv")

# -- Sweet viz save ( by PATH_OUTPUT ) 
feature_config = sv.FeatureConfig(skip=[],                                          # 要忽略哪個特徵
                                  force_cat=[], # Categorical特徵
                                  force_num=[],                                     # Numerical特徵
                                  force_text=None
                                 )                                                  # Text特徵

report_train_with_target_beftrain = sv.analyze([df_train, 'train'],
                                     target_feat = CONF.col_y, # 加入特徵變數
                                     feat_cfg = feature_config,
                                     pairwise_analysis = "off" # pairwise 2million, 200 column 預估要 37小時
                                     )
report_train_beftrain = sv.analyze([df_train, 'train'],
                         feat_cfg=feature_config,
                         pairwise_analysis = "off" # pairwise 2million, 200 column 預估要 37小時
                         )
report_train_with_target_beftrain.show_html(filepath = PATH_OUTPUT + 'Sweetviz_report_with_target(before_train).html')
report_train_beftrain.show_html(filepath = PATH_OUTPUT + 'Sweetviz_report_columns(before_train).html')

print("====Before train explotary done, path is <{0}>====".format(PATH_OUTPUT))




# %% [markdown]
# ##################################################################
# ## - Model Training 
# ##################################################################

# %%
# -- MLfow setting (auto log parameter)
# mlflow.xgboost.autolog()


# %%
# -- Parameter setting
# (note) [CONF.params] Parameter setting
# (note) [CONF.model_set] model description and algorithmn setting
print(CONF.params)
print(CONF.model_set)
model_set = model_trn.ModelSetting(
    model_name=CONF.model_set["model_name"],
    params=CONF.params,
    algo=CONF.model_set["algo"],
    metric=CONF.model_set["metric"],
    MODEL_PATH=PATH_OUTPUT + "trained_model/",
    tunning=False,
    reg=True
)


# %%
# -- Model Train
# (note) [CONF.col_y] [CONF.columns_valid_keep] [CONF.columns_drop]
print(CONF.col_y)
print(CONF.columns_valid_keep)
print(CONF.columns_drop)
df_valid_result = model_trn.model_train_cv(
    data_x=df_train,
    data_y=df_train[CONF.col_y],
    column_valid_keep=CONF.columns_valid_keep,
    columns_drop=CONF.columns_drop,
    model_setting=model_set,
    output_path=PATH_OUTPUT + "trained_model/",
    K_FOLD=5,
    col_groups_name=None)


# %% [markdown]
# # Record Track ==========================================================

# %% [markdown]
# #################################################################
# ## - ML flow Setting and Saving
# #################################################################

# %%
# -- Experiment name and paths setting (name, PATH_TRACK )
experiment_name = 'Train'
mlflow.tracking.set_tracking_uri(uri = PATH_TRACK )
try :
    mlflow.create_experiment(experiment_name)
    print("create experiment name " + experiment_name)
except :
    mlflow.set_experiment(experiment_name)
    print("set experiment name " + experiment_name)

# %%
# -- Calculate valuation 
# (Note) [CONF.col_y] 
fpr, tpr, thresholds = metrics.roc_curve(df_valid_result[ CONF.col_y ].to_list(), 
                                         df_valid_result[ "pred" ].to_list(), pos_label = 1)

# -- Log files to mlflow experiment (by metrics and PATH_OUTPUT)
with mlflow.start_run(experiment_id=1, run_name = CONF.model_set["model_name"]) as run:

    mlflow.log_metrics({ 
                        'auc' : metrics.auc(fpr, tpr)
    })
    mlflow.log_params(CONF.params)
    mlflow.log_artifact(PATH_OUTPUT)

# %% [markdown]
# #################################################################
# ## - Run ui
# #################################################################

# %%
import subprocess
subprocess.run(["mlflow", "ui","--backend-store-uri","proj_track","--host","0.0.0.0"])

# %%
"mlflow ui --backend-store-uri proj_track --host 0.0.0.0"

# %% [markdown]
# # Test (Testing functions and main) ===========================

# %% [markdown]
# # Explotary ================================
