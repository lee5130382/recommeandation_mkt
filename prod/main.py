"""
<預測程式>
    Author:
        陳彥霖
    Modified_Date:
        2023/05/15
    Model Version:
        1.0.0
    Model Type:
        LightGBM
    Introduction:
        Predict the purchasing potential of customer
    Step:
        1.ABT Preprocessing (encoding、assign data type)
        2.Predicting/Backtesting
            2-1.load model & load feature
            2-2.predict data
            2-3.ranking the prediction and create 'rank' column
     Ouput (main): 
        The prediction of data
"""

# -- package
# memory release
import gc
# System Package import
import os
# Cleaning
import pandas as pd
# time tracking
from tqdm import tqdm
# Packages Reloading ( Custom )
from func_pub.tools import chunk_load
from func_pub.logger import logger
from func_pub.dataprocessing import transf_to_category, output_clean_abt
from func_pub.predictor import Predictor
from prod.func.mkt_table import generate_mkt_table
from prod.configure import Configure

# -- path
PATH_DATA_TYPE = 'data/data_type/'
PATH_DATA_ATTR = 'data/data_attr/'
PATH_RAW_DATA = 'data/raw_data/'
PATH_ABT_DATA = 'data/model_abt/'
PATH_OUTPUT = 'dev/main/output/'
PATH_TEMP = 'data/temp/'
PATH_TRACK = 'proj_track/'

# -- configure
CONF = Configure()

# --------------------------------------
# -- 1. ABT Preprocessing (encoding、assign data type)
logger.info(f'(predict)ABT Preprocessing...')
if not(os.path.exists(f'{PATH_ABT_DATA}{CONF.season_date}{CONF.times}predict_ABT.csv')):
    # abt preprossing function
    data = output_clean_abt(dtype_dict_file=f'{PATH_DATA_TYPE}dtypes_20220825.csv',
                            raw_abt_path=PATH_RAW_DATA,
                            raw_abt_file=CONF.pred_data_name)
    data.to_csv(f'{PATH_ABT_DATA}{CONF.season_date}{CONF.times}predict_ABT.csv', index=False)
else:
    logger.info('已經有ABT.故可忽略跑ABT步驟')
logger.info(f'(predict)Complete ABT Preprocessing...')

# --------------------------------------
# -- 2. Predicting
logger.info(f'(predict)Start Predict Preprocessing')

# Create a dataframe to store the model prediction
df_pred_alltype = pd.DataFrame()
# Predict loop for multiple insurance product
for insur_type in tqdm(CONF.target_y):
    # 若前面流程中斷，從此處開始，則此防呆就可以起到作用
    if not('data' in locals()):
        data = chunk_load(path=PATH_ABT_DATA,
                          file=f'{CONF.season_date}{CONF.times}predict_ABT.csv',
                          size=CONF.chunk_size)
        # Transform variables to category type in the specified columns
        data = transf_to_category(data=data, path=PATH_DATA_ATTR)

    # predict
    dict_path = {'path_output':PATH_OUTPUT, 'path_track':PATH_TRACK}
    PRED = Predictor(insur_type, data, CONF, dict_path)
    df_pred_alltype = PRED.predict(df_pred_alltype, mode='predict')
    gc.collect()

# save prediction result
df_pred_alltype.to_csv(f'{PATH_TRACK}{CONF.season_date}predict_all_prediction.csv', index=False)

# --------------------------------------
# 檢查分布是否有過度集中(檢視min、max)
print(df_pred_alltype.shape)
df_pred_alltype.describe()

# --------------------------------------
# -- 3. generating mkt score table
generate_mkt_table()
