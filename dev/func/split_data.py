'''
split data
'''
# memory release
import gc
# Model - split Data
from sklearn.model_selection import train_test_split
# Packages Reloading ( Custom )
from func_pub.logger import logger
from func_pub.tools import chunk_load
from func_pub.dataprocessing import transf_to_category
from dev.main.configure import Configure
# change word color
from colorama import init, Fore, Style
init()

# configure
CONF = Configure()
# -- Basic paths Setting
PATH_DATA_ATTR = 'data/data_attr/'

def data_split(abt_path, abt_file, insur_type, save_test_file):
    '''Spliting data into training data and testing data
    
    Args : 
    -------
        abt_path : str
            The path of data
        abt_file : str
            File name
        insur_type : str
            Select the product type of insurance
        save_test_file : str
            File name

    Returns : 
    -------
        df_train : pandas.dataframe
            The training data. Annotation: testing have been stored.
    '''
    # 若前面流程中斷，從此處開始，則此防呆就可以起到作用
    if not('train' in locals()):
        train = chunk_load(path=abt_path, file=abt_file, size=CONF.chunk_size)
        # 將指定變數轉換為category型態
        train = transf_to_category(data=train, path=PATH_DATA_ATTR)

    # 移除-不須使用的Y，保留-當下要訓練的Y
    del_columns_y = ['Y_HOS', 'Y_SUR', 'Y_REI', 'Y_ACC', 'Y_DD', 'Y_LTC', 'Y_LIFE', 'Y_QUA']
    del_columns_y.remove(f'Y_{insur_type}')
    train.drop(columns=del_columns_y, axis=1, inplace=True)

    # 針對 Y 比例做分層抽樣 並區分 training / testing data
    df_train, df_test = train_test_split(train, test_size=0.2, stratify=train[f'Y_{insur_type}'],
                                         random_state=1020) # 0.8/0.2切分, seed: 1020

    # 檢查training and testing y=1 的占比
    train_y_percentage = round(df_train[f'Y_{insur_type}'].mean()*100, 5)
    test_y_percentage = round(df_test[f'Y_{insur_type}'].mean()*100, 5)
    if abs(train_y_percentage - test_y_percentage) > 0.01:
        logger.error(f'Spliting Fail at the {insur_type} flow')
        logger.error(f'''Have a difference of the rate of Y between trainset and testset.
        Seperately, train:{train_y_percentage} and test:{test_y_percentage}''')
        raise ValueError(
            f'''{Fore.RED}Have a difference of the rate of Y between trainset and testset. 
            Seperately, train:{train_y_percentage} and test:{test_y_percentage}{Style.RESET_ALL}''')

    logger.info(f'Spliting Successful at the {insur_type} flow')
    del train
    gc.collect()

    ## 將testdata存到暫存區
    df_test.to_csv(save_test_file, index=False)
    del df_test
    gc.collect()

    return df_train
