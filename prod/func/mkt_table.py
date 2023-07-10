"""
<保障型行銷分>
    Author:
        陳彥霖
    Modified_Date:
        2023/03/08
    Version:
        1.0.0
    Introduction:
        Generate marketing score table (a.k.a MKT)
    Step:
        1.Establishing the level/divid by the rank of model prediction 
          and adjust the length of probability column
        2.Adjusting the columns order to match the subsequent HAP and TD data table column order for deployment 
        3.Generating the predict data and tag data (automatic)
        4.Verifying the data table
"""
# -- Import package for mkt score
# System Package import
import os
import sys
# Cleaning
import numpy as np
import pandas as pd
# Packages Reloading ( Custom )
from prod.configure import Configure
from func_pub.logger import logger
# change word color
from colorama import init, Fore, Style
init()


# -- ROOT paths setting (Default to project root dir)
ROOT = 'D:/LEE/BOX_ALL/BOX_NEW/行銷分ABT與模型檔/正式模型區/保障型行銷分GIT/model'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
logger.info(os.getcwd())
logger.info(sys.path)

# -- Basic paths Setting
PATH_TRACK = 'proj_track/'
# Set up time
pred_date = '2023-05-15' # follow 基底表時間

# -- Call parameter from configure
CONF = Configure()

def generate_mkt_table():
    '''
    Generate marketing score table
    '''
    # --------------------------------------
    # -- 1.Establishing the level/divid by the rank of model prediction and adjust the length of probability column

    # Load Data
    prediction_data = pd.read_csv(f'{PATH_TRACK}{CONF.season_date}_all_prediction.csv')
    for name in CONF.target_y:
        # 分成10檻，3以後換成空值
        prediction_data[f'{name}_MKT_LEVEL'] = pd.qcut(prediction_data[f'{name}_MKT_RANK'], 10,
                                                       labels=False) + 1
        prediction_data.loc[prediction_data[f'{name}_MKT_LEVEL'] > 3, f'{name}_MKT_LEVEL'] = np.nan
        # 分成1000檻
        prediction_data[f'{name}_MKT_DIVID'] = pd.qcut(prediction_data[f'{name}_MKT_RANK'], 1000,
                                                       labels=False) + 1
        # _MKT_PROB 取四捨五入至15位
        prediction_data[f'{name}_MKT_PROB'] = round(prediction_data[f'{name}_MKT_PROB'], 15)
        # 回測促約 (暫定空值)
        prediction_data[f'{name}_MKT_P_RATE'] = " "

    # --------------------------------------
    # -- 2.Adjusting the columns order to match the subsequent HAP and TD data table column order for deployment
    final = prediction_data[["ID",
                            "HOS_MKT_RANK","HOS_MKT_LEVEL","HOS_MKT_DIVID","HOS_MKT_P_RATE",
                            "SUR_MKT_RANK","SUR_MKT_LEVEL","SUR_MKT_DIVID","SUR_MKT_P_RATE",
                            "REI_MKT_RANK","REI_MKT_LEVEL","REI_MKT_DIVID","REI_MKT_P_RATE",
                            "ACC_MKT_RANK","ACC_MKT_LEVEL","ACC_MKT_DIVID","ACC_MKT_P_RATE",
                            "DD_MKT_RANK","DD_MKT_LEVEL","DD_MKT_DIVID","DD_MKT_P_RATE",
                            "LTC_MKT_RANK","LTC_MKT_LEVEL","LTC_MKT_DIVID","LTC_MKT_P_RATE",
                            "LIFE_MKT_RANK","LIFE_MKT_LEVEL","LIFE_MKT_DIVID","LIFE_MKT_P_RATE",
                            "HOS_MKT_PROB","SUR_MKT_PROB","REI_MKT_PROB","ACC_MKT_PROB",
                            "DD_MKT_PROB","LTC_MKT_PROB","LIFE_MKT_PROB"
                            ]]

    # --------------------------------------
    # -- 3.Generating the predict data and tag data (automatic)
    # predict date
    final['PREDICT_ABT_DATE'] = pred_date # PREDICT_ABT_DATE
    # tag date
    # 列出貼標月份
    month_list = range(int(CONF.season_date.split('_')[1]), int(CONF.season_date.split('_')[2])+1)
    year = int(CONF.season_date.split('_')[0]) # 列出貼標年份
    month = month_list[int(CONF.times.split('_')[1]) - 1] # 抓取第X個月份
    final['TAG_DATE'] = f'{str(year)}-{str(month).zfill(2)}-01' #12/20調整，原為str(month)，個位數月份無補0
    final.rename(columns = {'ID': 'SAS_ID'}, inplace=True)
    logger.info(f'資料筆數為 {final.shape[0]} & 欄位數為 {final.shape[1]}')


    # --------------------------------------
    # -- 4.Verifying the data table
    '''
    排名(_MKT_RANK): 數值型,1-總樣本數，無空值。
    等級貼標(_MKT_LEVEL): 數值型,1、2、3,有空值。(欄位定義: 分別為前10%、11-20%、21-30%,排序低於31%客戶不貼標。)
    分檻(_MKT_DIVID): 數值型,1-1000,無空值(貼1-1000,分別為前0.1%、0.2%、0.3%...至100%。)
    回測促約率(_MKT_P_RATE): (暫無)。
    模型貼標值(_MKT_PROB): 數值型,至多15位小數點,介於0-1。
    '''
    for name in CONF.target_y:
        # 判斷資料型別
        if (final[f'{name}_MKT_RANK'].dtype == 'float64'
            and final[f'{name}_MKT_LEVEL'].dtype == 'float64'
            and final[f'{name}_MKT_DIVID'].dtype == 'int64'
            and final[f'{name}_MKT_PROB'].dtype == 'float64'):
            logger.info(f'險種: {name} 欄位型別為: 數值型，欄位型別正確')
        else:
            logger.info(f'{Fore.RED}險種: {name} 欄位型別錯誤{Style.RESET_ALL}')
            break
        # 判斷欄位內容
        if (final[f'{name}_MKT_RANK'].max() == len(prediction_data)
            and sorted(list(final[f'{name}_MKT_LEVEL'].dropna().unique())) == [1,2,3]
            and len(final[f'{name}_MKT_DIVID'].unique()) == 1000
            and (final.loc[ final[f'{name}_MKT_PROB'] == final[f'{name}_MKT_PROB'].max(), f'{name}_MKT_LEVEL'] == 1).bool()
            and final[f'{name}_MKT_PROB'].max() < 1):
            logger.info('Correct:欄位內容正確')
        else:
            logger.info(f'{Fore.RED}險種: {name} 欄位內容錯誤{Style.RESET_ALL}')
            break
        # 空值判斷
        if (final[f'{name}_MKT_RANK'].isnull().sum() == 0
            and final[f'{name}_MKT_LEVEL'].isnull().sum() > 0
            and final[f'{name}_MKT_DIVID'].isnull().sum() == 0
            and final[f'{name}_MKT_PROB'].isnull().sum() == 0
            and final['PREDICT_ABT_DATE'].isnull().sum() == 0
            and final['TAG_DATE'].isnull().sum() == 0):
            logger.info('Correct:欄位空值正確')
        else:
            logger.info(f'{Fore.RED}險種: {name} 欄位空值異常{Style.RESET_ALL}')
            break

    # 存放行銷分大表
    final.to_csv(f'{PATH_TRACK}IFCUSDPDT0001_{str(year)}{str(month).zfill(2)}_nonrevise.csv',
                 index=False)
