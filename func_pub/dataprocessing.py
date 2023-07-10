"""
The tools of the data processing
"""
# import packages
import gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# change word color
from colorama import init, Fore, Style
init()
from func_pub.tools import chunk_load
from func_pub.logger import logger
from dev.main.configure import Configure

# -- Basic paths Setting
PATH_DATA_ATTR = 'data/data_attr/'

# -- Call parameter from configure
CONF = Configure()

# collect columns
def get_columns(data, columns_type):
    '''Select a list of variables that match the data format 
    
    Args : 
    -------
        data : pandas.DataFrame
            Only dataframe 
        columns_type : str
            Any type of variable. For example: object、int、float 

    Returns : 
    -------
        list_match_col : list
            List of variables that match the specified data format
    '''
    list_match_col = []
    for i in data.columns:
        if data[i].dtype == columns_type:
            list_match_col.append(i)
    return list_match_col

# label encoding
def label_encoding(data, columns):
    '''Label encoding the specified columns
    
    Args : 
    -------
        data : pandas.DataFrame
            Only dataframe 
        columns_type : str
            List of variables

    Returns : 
    -------
        None. This function is a process without output
    '''
    labelencoder = LabelEncoder()
    for i in columns:
        data[i] = data[i].fillna("MM") # 進行label encoding 不能有空值
        data[i] = labelencoder.fit_transform(data[i])

# transform variable's type to category
def transf_to_category(data, path):
    '''Select a list of variables that match the data format 
    
    Args : 
    -------
        data : pandas.DataFrame
            Only dataframe 
        path : str
            Input the path where store the file called '變數紀錄表_SQL用_0824.csv'

    Returns : 
    -------
        data : pandas.DataFrame
            The data where some of part have been adjusted to categorical type
    '''
    # 匯入類別變數判定表
    df_sql = pd.read_csv(f'{path}變數紀錄表_SQL用_0824.csv', encoding='cp950')
    # 檢視是否有變數不一致
    del_var = ['ID', 'FIRE_POLICY_NO_SUM_R', 'CAR_POLICY_NO_NUM_R',
               'Y_QUA', 'Y_HOS', 'Y_SUR', 'Y_REI', 'Y_ACC', 'Y_DD', 'Y_LTC', 'Y_LIFE']
    var = [ i for i in data.columns if i not in (list(df_sql['變數名稱']) + del_var) ]
    if len(var) > 0:
        raise ValueError(f'The columns in the abt is not in the 變數紀錄表, including: {var}')

    # 利用類別變數判定表，抓出需指定為類別型態之變數，再排除包含IND跟FLG之變數(不須轉為CATEGORY)
    categorical_feature =  list(df_sql.loc[df_sql['欄位類別'] == 'Categorical', '變數名稱'])
    categorical_feature_r = [ i for i in categorical_feature if ('IND' not in i) & ('FLG' not in i)]
    categorical_feature_r.remove('CUST_CLASS_CD')
    del df_sql
    for i in categorical_feature_r:
        data[i] = data[i].astype('category')
    return data

def output_clean_abt(dtype_dict_file, raw_abt_path, raw_abt_file):
    '''Data processing including label encoding
    
    Args : 
    -------
        dtype_dict_file : str
            File name. The file contains column name and column format 
        raw_abt_path : str
            The path of raw abt data
        raw_abt_file : str
            New file name for abt

    Returns : 
    -------
        data : pandas.dataframe
            New abt through data preprocessing
    '''
    ## loading traindata
    # load type setting file
    dtype_dict_data = pd.read_csv(dtype_dict_file, names=['column', 'coltype'])
    dtype_dict = dict(zip(dtype_dict_data.column, dtype_dict_data.coltype))
    # use chunk to load data
    data = chunk_load(path=raw_abt_path, file=raw_abt_file,
                      size=CONF.chunk_size, dtype_dict=dtype_dict) # dtype_dict : 讀檔指定欄位型態

    # 移除不穩定變數(後續應由ABT製作檔校正之)
    data.drop('CUST_CLASS_CD', axis=1, inplace=True)

    # 紀錄原始欄位數量，以利後續防呆，檢視欄位是否異常增加
    raw_columns_cnt = len(data.columns)

    ## Feature Engineering
    # 將指定變數轉換為category型態
    data = transf_to_category(data=data, path=PATH_DATA_ATTR)
    # 先移除ID以利進行encdoning
    training_data_id = data.pop(CONF.cust_id)
    # label encoding
    object_columns = get_columns(data, 'object')
    label_encoding(data, object_columns)

    ## 將原本的ID欄位串回來
    data[CONF.cust_id] = training_data_id
    del training_data_id, object_columns
    gc.collect()

    ## 檢查並存放
    if len(data.columns) != raw_columns_cnt:
        logger.error('ABT跑檔失敗')
        logger.error(f'The number of column of abt is incorrect : {len(data.columns)}')
        raise ValueError(f'{Fore.RED}The number of column of abt is incorrect : {len(data.columns)}{Style.RESET_ALL}')

    logger.info('ABT跑檔完成')
    return data
