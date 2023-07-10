"""
<保障型行銷分>
    Author:
        陳彥霖
    Modified_Date:
        2023/05/15
    Version:
        1.1.0
    Introduction:
        Use tool 'Sweetviz' to check if the data between train and backtest/test have different distriubtion
    Step:
        1.Load train abt
        2.Load pred abt & Running Sweetviz
"""

# import package
# System Package import
import os
import sys
import sweetviz as sv
import pandas as pd
import numpy as np
from datetime import datetime
# Packages Reloading ( Custom )
from func_pub.tools import chunk_load

# -- ROOT paths setting (Default to project root dir)
ROOT = 'D:\LEE\BOX_ALL\BOX_NEW\行銷分ABT與模型檔\正式模型區\保障型行銷分GIT\model'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
print(os.getcwd())
print(sys.path)

# -- Basic paths Setting
PATH_DATA_TYPE = 'data/data_type/'
PATH_RAW_DATA = 'data/raw_data/'
PATH_TRACK = 'proj_track/'

# -- setting
target_y = ['HOS', 'SUR', 'REI', 'ACC', 'DD', 'LTC', 'LIFE']
chunk_size = 100000
mode = 'predict'
train_data_name = 'TRAIN_ABT_COLS_20221124.csv'
pred_data_name = 'MKT_COMB_COLS_20230515.csv'

# --------------------------------------
# -- 1. Load train abt
# load type setting file
dtype_dict_file = f'{PATH_DATA_TYPE}dtypes_20220825.csv'
dtype_dict_data = pd.read_csv(dtype_dict_file, names=['column', 'coltype'])
dtype_dict = dict(zip(dtype_dict_data.column, dtype_dict_data.coltype))
# use chunk to load data
df_train = chunk_load(path=PATH_RAW_DATA, file=train_data_name,
                      size=chunk_size, dtype_dict=dtype_dict) # dtype_dict : 讀檔指定欄位型態

# --------------------------------------
# -- 2.Load pred abt & Running Sweetviz
# feature_config = sv.FeatureConfig(force_cat = ['IF_CREDIT_FLG','CREDIT_CATHAYBK_FLG']) # 處理error
time = datetime.now().strftime('%Y-%m-%d')
if mode == 'backtest':
    # Load pred abt
    df_pred = chunk_load(path=PATH_RAW_DATA, file=f'TRAIN_ABT_COLS_{pred_data_name}.csv',
                         size=chunk_size, dtype_dict=dtype_dict) # dtype_dict : 讀檔指定欄位型態
    # Running Sweetviz
    compare_report = sv.compare([df_train.drop(['ID'], axis=1), 'Training Data'],
                                [df_pred.drop(['ID'], axis=1), 'Backtest Data'],
                                pairwise_analysis='off') # target_feat='Y_HOS',
    compare_report.show_html(filepath=f'{PATH_TRACK}Compare_train_backtest_report_{time}.html')
else:
    # Load pred abt
    df_pred = chunk_load(path=PATH_RAW_DATA, file=pred_data_name,
                         size=chunk_size, dtype_dict=dtype_dict) # dtype_dict : 讀檔指定欄位型態
    # Running Sweetviz    
    compare_report = sv.compare([df_train.drop(['ID','Y_QUA']+[f'Y_{i}' for i in target_y], axis=1), 'Training Data'],
                                [df_pred.drop(['ID'], axis=1), 'Test Data'], pairwise_analysis='off')
    compare_report.show_html(filepath=f'{PATH_TRACK}Compare_train_test_report_{time}.html')


