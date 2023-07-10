'''
Other tools that cannot be grouped
'''
# -- Import package
import pandas as pd
from tqdm import tqdm
import os
from func_pub.logger import logger


# create folder
def create_folder(path, folder_name):
    if not os.path.exists(os.path.join(path, folder_name)):
        os.makedirs(os.path.join(path, folder_name))
        logger.info(f'create folder named {folder_name}')

def calulate_buy_rate(data, insur_type, percentage):
    '''calculating the purchase rate and times
    
    Args : 
    -------
        data : pandas.DataFrame
            Only dataframe
        insur_type : str
            Select the product type of insurance 
        percentage : float
            Should be between 0.0 and 1.0 and represent the proportion of dataset will be trimmed 
            in order to calculate the purchase rate

    Returns : 
    -------
        buy_rate : float
            the purchase rate of data
        buy_rate_times : int
            the purchase times of data
    '''
    # 依照該險種RANK排序
    buy_rate_rank = data.sort_values([f'{insur_type}_MKT_RANK'], ascending=True)
    # 計算不同比例的促約率
    buy_rate = round(buy_rate_rank.iloc[range(int(len(data)*percentage))][f'Y_{insur_type}'].mean(), 4)
    # 計算促約倍數
    buy_rate_times = round(buy_rate/data[f'Y_{insur_type}'].mean(), 4)
    return buy_rate, buy_rate_times

### use chunk to load data
def chunk_load(path, file, size, dtype_dict = None):
    '''Loading data by the chunk method
    
    Args : 
    -------
        path : str
            Path of data
        file : str
            File name
        size : int
            Any interger, But not 0. Represent the size of every chunk of the data
        dtype_dict : dictionary, default=None
            Dictionary contains column name and column format. Ex.{age:int8}

    Returns : 
    -------
        data : pandas.dataframe
    '''
    data_chunk = pd.read_csv(f'{path}{file}', encoding="CP950", dtype=dtype_dict, chunksize=size)
    data_temp = []
    for chunk in tqdm(data_chunk):
        data_temp.append(chunk)
    data = pd.concat(data_temp, axis=0)
    del data_temp, data_chunk
    return data
