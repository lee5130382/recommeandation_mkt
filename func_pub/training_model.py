'''
training model flow including training phase, predict validation and save model pickle
'''
# memery lease
import gc
# data process
import pandas as pd
# Storing Tool
import joblib
# Model - LightGBM
import lightgbm as lgb
# Model - Performance
from sklearn.metrics import roc_auc_score
# K-fold
from sklearn.model_selection import StratifiedKFold
# time tracking
from tqdm import tqdm
# import custom function
from func_pub.logger import logger
# configure
from dev.main.configure import Configure
CONF = Configure()

# model train
def train_model(data, insur_type, verbose_eval=True, writer=None):
    '''Training 5-fold model and storing the AUC of train dataset and validation dataset 
    
    Args : 
    -------
        data : pandas.dataframe
            Only dataframe
        insur_type : str
            Select the product type of insurance 
        verbose_eval : bool or int, optional (default=True)
            Requires at least one validation data.
            If True, the eval metric on the valid set is printed at each boosting stage.
            If int, the eval metric on the valid set is printed at every ``verbose_eval`` boosting stage.
            The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed.
        writer : writer, default=None
            Input writer object to store the outcome

    Returns : 
    -------
        features : list
            List of features
        df_imp_feature : pandas.dataframe
            Dataframe of features and importance with k-fold
        lgbm_original : Booster
            The trained Booster model

    '''
    ## 模型訓練前置作業(先預留輸出空間、切k-fold、轉換模型可解讀模式)
    # 儲存訓練特徵
    features = [col for col in data.columns if col not in [CONF.cust_id, f'Y_{insur_type}']]

    ## 預先設定好欲產出之欄位
    # 先存train資料裡的id&y欄位
    oof = data.loc[:, [CONF.cust_id, f'Y_{insur_type}']]
    # 先預留2個欄位給train、valid預測值
    oof['train_pred'] = 0
    oof['valid_pred'] = 0
    # 預留5個list裡面都是0，以利後續存放不同fold的模型結果
    lgbm_original = [0]*CONF.fold_num
    # 預留一個df，蒐集變數重要性
    df_imp_feature = pd.DataFrame()

    ## 切fold訓練
    kfold = StratifiedKFold(n_splits=CONF.fold_num, shuffle=True, random_state=1020)
    ## 將ABT轉換成array方便進行K-fold取檔
    df_train_x = data[features].values
    df_train_y = data[f'Y_{insur_type}'].values
    del data
    gc.collect()

    ## 模型訓練
    for fold, (trn_idx, val_idx) in tqdm(enumerate(kfold.split(df_train_x, df_train_y))):
        x_train, y_train = df_train_x[trn_idx], df_train_y[trn_idx]
        x_valid, y_valid = df_train_x[val_idx], df_train_y[val_idx]
        # 將資料轉成lightGBM的型式做模型訓練
        trn_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_valid, label=y_valid)
        # lgbm documentation: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        lgbm_original[fold] = lgb.train(CONF.lgb_params,
                                        trn_data,
                                        categorical_feature='auto',
                                        valid_sets=[trn_data, val_data],
                                        verbose_eval=verbose_eval
                                        )
        # 留下變數importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = lgbm_original[fold].feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold + 1
        # 疊加5個fold的重要變數
        df_imp_feature = pd.concat([df_imp_feature, fold_importance_df], axis=0)
        # training的預測值
        oof['train_pred'].iloc[trn_idx] = lgbm_original[fold].predict(x_train)
        # validation的預測值
        oof['valid_pred'].iloc[val_idx] = lgbm_original[fold].predict(x_valid)
        # 檢視5個fold的train auc、valid auc
        train_auc = roc_auc_score(oof.iloc[trn_idx][f'Y_{insur_type}'],
                                  oof.iloc[trn_idx]['train_pred'])
        valid_auc = roc_auc_score(oof.iloc[val_idx][f'Y_{insur_type}'],
                                  oof.iloc[val_idx]['valid_pred'])
        logger.info(f'{insur_type} train auc of {fold+1}th fold : {round(train_auc, 4)}')
        logger.info(f'{insur_type} valid auc of {fold+1}th fold : {round(valid_auc, 4)}')
        # 儲存進DF，以利後續匯出CSV留存訓練AUC
        writer.writerow([insur_type, fold, 'train', train_auc])
        writer.writerow([insur_type, fold, 'valid', valid_auc])
        del x_train, y_train, x_valid, y_valid
        gc.collect()
    del df_train_x, df_train_y
    gc.collect()

    return features, df_imp_feature, lgbm_original


# model predict
def predict_result(data, features, lgbm, fold_num, insur_type):
    '''Make predictions using the pretrained model from the previous step
    
    Args : 
    -------
        data : pandas.dataframe
            Only dataframe
        features : list
            List of features required for model predict
        lgbm : Booster
            The trained Booster model
        fold_num : int
            fold_num must be the same as the number of fold from the pretrained model
        insur_type : str
            Select the product type of insurance

    Returns : 
    -------
        predictions : pandas.dataframe
            The predicted values for different folds and the average predicted values for all folds

    '''
    predictions = data[[CONF.cust_id]] # 貼回ID
    x_test1 = data[features].values
    for i in range(fold_num):
        predictions[f'fold{i+1}'] = lgbm[i].predict(x_test1) # 預測
    # 預測值(k-fold)取平均
    predictions['mean_'+insur_type] = predictions[[i for i in predictions.columns if i.startswith('fold')]].mean(axis=1)
    del x_test1
    gc.collect()
    return predictions

# save model
def save_model(route, filename, lgbm , fold):
    '''Saving model with pickle format
    
    Args : 
    -------
        route : str
            Path
        filename : str
            File name
        lgbm : Booster
            The trained Booster model
        fold : int
            fold_num must be the same as the number of fold from the pretrained model

    Returns : 
    -------
        None. This function is a process without output

    '''
    for i in range(fold):
        joblib.dump(lgbm[i],
                    f'{route}{filename}_{i+1}.pkl')
