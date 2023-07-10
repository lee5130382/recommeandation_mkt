"""
<回測程式>
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
        2.Backtesting
            2-1.load model & load feature
            2-2.predict data 
            2-3.rank the prediction and create 'rank' column
        3.Storing the auc, lift, buy rate times 
     Ouput (main): 
        The prediction of data and Auc, lift, buy rate times 
"""

# -- package
# other
import csv
# memory release
import gc
# System Package import
import os
# Cleaning
import pandas as pd
# Model - Performance
from sklearn.metrics import roc_auc_score
# time tracking
from tqdm import tqdm
# Packages Reloading ( Custom )
from func_pub.tools import chunk_load, calulate_buy_rate
from func_pub.logger import logger
from func_pub.dataprocessing import output_clean_abt, transf_to_category
from func_pub.predictor import Predictor
from dev.main.configure import Configure

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


def backtest():
    '''
    backtest flow
    '''
    # --------------------------------------
    # -- 1. ABT Preprocessing (encoding、assign data type)
    logger.info(f'(backtest)ABT Preprocessing...')
    if not (os.path.exists(f'{PATH_ABT_DATA}{CONF.season_date}backtest_ABT.csv')):
        # abt preprossing function
        data = output_clean_abt(dtype_dict_file=f'{PATH_DATA_TYPE}dtypes_20220825.csv',
                                raw_abt_path=PATH_RAW_DATA,
                                raw_abt_file=CONF.backtest_data_name)
        data.to_csv(
            f'{PATH_ABT_DATA}{CONF.season_date}backtest_ABT.csv', index=False)
    else:
        logger.info('已經有ABT.故可忽略跑ABT步驟')
    logger.info(f'(backtest)Complete ABT Preprocessing...')

    # --------------------------------------
    # -- 2. Backtesting
    logger.info(f'(backtest)Start Predict processing')

    # Create a dataframe to store the model prediction
    df_pred_alltype = pd.DataFrame()
    # Predict loop for multiple insurance product
    for insur_type in tqdm(CONF.target_y):
        # 若前面流程中斷，從此處開始，則此防呆就可以起到作用
        if not ('data' in locals()):
            data = chunk_load(path=PATH_ABT_DATA,
                              file=f'{CONF.season_date}backtest_ABT.csv',
                              size=CONF.chunk_size)
            # Transform variables to category type in the specified columns
            data = transf_to_category(data=data, path=PATH_DATA_ATTR)

        # predict
        dict_path = {'path_output': PATH_OUTPUT, 'path_track': PATH_TRACK}
        pred = Predictor(insur_type, data, CONF, dict_path)
        df_pred_alltype = pred.predict(df_pred_alltype, mode='backtest')
        del data
        gc.collect()

    # save prediction result
    df_pred_alltype.to_csv(f'{PATH_TRACK}{CONF.season_date}backtest_all_prediction.csv',
                           index=False)
    logger.info(f'(backtest)Complete Predict processing')
    # --------------------------------------
    # -- 3. Storing the auc, lift, buy rate times
    # 創建並開啟CSV寫檔，以利邊跑邊存放所有的AUC、LIFT值
    header = ['INSUR_TYPE', 'DATASET', 'AUC',
              'buy_rate_1', 'buy_times_1', 'buy_rate_2', 'buy_times_2', 'buy_rate_3', 'buy_times_3']
    file_pred = open(
        f'{PATH_TRACK}backtest_model_history.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file_pred)
    writer.writerow(header)  # write the header
    
    logger.info(f'(backtest)running model index (auc、times)')
    # 跑各險種AUC、LIFT值
    for insur_type in CONF.target_y:
        pred_auc = roc_auc_score(df_pred_alltype[f'Y_{insur_type}'],
                                 df_pred_alltype[f'{insur_type}_MKT_PROB'])  # 計算各險種AUC

        # 計算前10/20/30%促約率，並計算寬鬆大盤促約率
        buy_rate_1, buy_times_1 = calulate_buy_rate(
            df_pred_alltype, insur_type, percentage=CONF.anal_rate[0])
        buy_rate_2, buy_times_2 = calulate_buy_rate(
            df_pred_alltype, insur_type, percentage=CONF.anal_rate[1])
        buy_rate_3, buy_times_3 = calulate_buy_rate(
            df_pred_alltype, insur_type, percentage=CONF.anal_rate[2])

        # LIFT圖
        # kds.metrics.plot_lift(
        #     data['Y_'+insur_type], data[insur_type+'_MKT_PROB'], title=insur_type+' LIFT PLOT')
        # kds.metrics.report(
        #     data['Y_'+insur_type], data[insur_type+'_MKT_PROB'], title=insur_type+' PLOT')
        # savefig()

        # 儲存進CSV
        writer.writerow([insur_type, backtest, pred_auc,
                        buy_rate_1, buy_times_1, buy_rate_2, buy_times_2, buy_rate_3, buy_times_3])

    # 終止寫入CSV
    file_pred.close()
    logger.info(f'finish the whole backtest flow')