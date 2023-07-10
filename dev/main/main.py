"""
<保障型行銷分>
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
        2.Training Model and testing model by valid data and test data  
            2-1.Spliting data to train / test data by Y distribution
            2-2.Training k-fold model (and validation)
            2-3.Using k-fold models to predict test data
            2-4.storing important features and model pickle file
        3.Backtesting the stablility of model
     Ouput (main): 
        modol pickle for k-fold
     Ouput (others): 
        feature importance list
        model criterion : AUC
"""

# -- jupyternotebook Setting (tab the package too slow (avoid memory exhausted))
# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

# -- Import package for mkt score
# System Package import
import csv
import os
import sys
# memory release
import gc
# Storing Tool
import pickle
# Other package
import warnings
# Model - Performance
from sklearn.metrics import roc_auc_score
# time tracking
from tqdm import tqdm
# Packages Reloading ( Custom )
from dev.main.configure import Configure
from dev.func.split_data import data_split
from dev.func.backtest import backtest
from func_pub.tools import chunk_load, create_folder
from func_pub.dataprocessing import transf_to_category, output_clean_abt
from func_pub.logger import logger
from func_pub.training_model import train_model, save_model
warnings.filterwarnings('ignore')

# -- ROOT paths setting (Default to project root dir)
ROOT = 'D:/LEE/BOX_ALL/BOX_NEW/行銷分ABT與模型檔/正式模型區/保障型行銷分GIT/model'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
print(os.getcwd())
print(sys.path)

# -- Create dir
dir_list = ['raw_data', 'model_abt', 'importance', 'temp']
for folder in dir_list:
    create_folder(path=os.path.join(ROOT, 'data'), folder_name=folder)
create_folder(path=os.path.join(ROOT, 'dev/main'), folder_name='output')
create_folder(path=os.path.join(ROOT), folder_name='proj_track')

# -- Basic paths Setting
PATH_DATA_TYPE = 'data/data_type/'
PATH_DATA_ATTR = 'data/data_attr/'
PATH_RAW_DATA = 'data/raw_data/'
PATH_ABT_DATA = 'data/model_abt/'
PATH_TEMP = 'data/temp/'
PATH_IMPORTANT = 'data/importance/'
PATH_OUTPUT = 'dev/main/output/'
PATH_TRACK = 'proj_track/'

# -- Call parameter from configure
CONF = Configure()

# --------------------------------------
# -- 1. ABT Preprocessing (encoding、assign data type)
logger.info('ABT Preprocessing...')
if not(os.path.exists(f'{PATH_ABT_DATA}{CONF.season_date}_TRAINMODEL_ABT.csv')):
    train = output_clean_abt(dtype_dict_file=f'{PATH_DATA_TYPE}dtypes_20220825.csv',
                             raw_abt_path=PATH_RAW_DATA,
                             raw_abt_file=CONF.train_data_name)
    train.to_csv(f'{PATH_ABT_DATA}{CONF.season_date}_TRAINMODEL_ABT.csv', index=False)
else :
    logger.info('已經有ABT.故可忽略跑ABT步驟')
logger.info('Complete ABT Preprocessing...')

# --------------------------------------
# -- 2. Training Model and testing model by valid data and test data
## 創建並開啟CSV寫檔，以利邊跑邊存放所有險種的train_auc、valid_auc、test_auc
header = ['INSUR_TYPE', 'FOLD', 'DATASET', 'AUC']
file = open(f'{PATH_TRACK}training_model_history.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(file)
# write the header
writer.writerow(header)

logger.info('Start Building Model processing')
## 跑7種險種的模型訓練
for insur_type in tqdm(CONF.target_y):

    # 2-1. Spliting data to train / test data by Y distribution
    df_train = data_split(abt_path=PATH_ABT_DATA,
                          abt_file=f'{CONF.season_date}_TRAINMODEL_ABT.csv',
                          insur_type=insur_type,
                          save_test_file=f'{PATH_TEMP}{insur_type}_testdata.csv')

    # 2-2. training k-fold model (and validation)
    features, df_imp_feature, lgbm_original = train_model(data=df_train, insur_type=insur_type,
                                                          verbose_eval=CONF.verbose_eval,
                                                          writer=writer)
    del df_train
    gc.collect()
    logger.info(f'5-fold training successful at the {insur_type} flow')

    # 2-3. Using k-fold models to predict test data
    # 重新匯入TEST資料
    df_test = chunk_load(path=PATH_TEMP, file=f'{insur_type}_testdata.csv', size=CONF.chunk_size)
    df_test = transf_to_category(data=df_test, path=PATH_DATA_ATTR)

    # 進行模型predict
    X_test = df_test[features].values
    df_predictions = df_test[[CONF.cust_id, f'Y_{insur_type}']]
    for fold in range(0, CONF.fold_num):
        df_predictions[f'fold{fold+1}'] = lgbm_original[fold].predict(X_test)
        test_auc = roc_auc_score(df_predictions[f'Y_{insur_type}'], df_predictions[f'fold{fold+1}'])
        writer.writerow([insur_type, fold, 'test', test_auc])

    # 平均5個模型的預測值
    df_predictions['mean'] = df_predictions[[i for i in df_predictions.columns if i.startswith('fold')]].mean(axis=1)
    mean_test_auc = roc_auc_score(df_predictions[f'Y_{insur_type}'], df_predictions['mean'])
    print(f'{insur_type} test auc of mean fold : {round(mean_test_auc, 4)}')
    writer.writerow([insur_type, 'mean', 'test', mean_test_auc])

    # 2-4. storing important features and model pickle file
    # 累加Importance，並存放importance
    # 依照fold排序
    df_imp_feature = df_imp_feature.sort_values(by=['fold', 'feature'], ascending=True)
    # groupby'feature'欄位累加importance值
    df_imp_feature['cumsum'] = df_imp_feature[['importance', 'feature']].groupby('feature').cumsum()
    df_imp_feature.to_csv(f'{PATH_IMPORTANT}{CONF.season_date}{insur_type}_重要特徵.csv', index = False)

    # 存放模型 與 pickle (分開存放5個模型pickle檔案)
    save_model(PATH_OUTPUT, f'{insur_type}_模型_{CONF.season_date}',
               lgbm_original, fold=CONF.fold_num)
    with open(f'{PATH_OUTPUT}{insur_type}_模型_{CONF.season_date}.pkl', "wb") as file2:
        pickle.dump(features, file2)
    del df_test, X_test, lgbm_original, df_imp_feature, features, df_predictions
    gc.collect()

logger.info('Complete Building Model processing')
# 終止寫入CSV
file.close()

# --------------------------------------
# -- 3. Backtesting the stablility of model
backtest()
