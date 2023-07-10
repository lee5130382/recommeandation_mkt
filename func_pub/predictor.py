'''
predict flow with predictive tools
'''
# memory release
import gc
# Storing Tool
import pickle
import joblib
# Packages Reloading ( Custom )
from func_pub.training_model import predict_result

class Predictor:
    def __init__(self, insur_type, data, configure, dict_path):
        self.insur_type = insur_type
        self.data = data
        self.conf = configure
        self.dict_path = dict_path

    def load_model(self, route, filename, fold):
        '''Loading model with pickle format
    
        Args : 
        -------
            route : str
                Path
            filename : str
                File name
            fold : int
                fold_num must be the same as the number of fold from the pretrained model

        Returns : 
        -------
            lgb : Booster
                The trained Booster model
            features : list
                The features of trained model
        '''
        lgb = [0]*fold
        for i in range(fold):
            lgb[i] = joblib.load(f'{route}{filename}_{i+1}.pkl')
        with open(f'{route}{self.insur_type}_模型_{self.conf.season_date}.pkl', "rb") as file_load_model:
            features = pickle.load(file_load_model)
        return lgb, features

    def retain_y(self):
        '''Retain Y which model needs
        '''
        del_columns_y = ['Y_HOS', 'Y_SUR', 'Y_REI', 'Y_ACC', 'Y_DD', 'Y_LTC', 'Y_LIFE', 'Y_QUA']
        del_columns_y.remove(f'Y_{self.insur_type}')
        self.data.drop(columns=del_columns_y, axis=1, inplace=True)

    def paste_y(self, data):
        '''insert Y to data
        
        Args : 
        -------
            data : dataframe
        '''
        data[f'Y_{self.insur_type}'] = self.data[f'Y_{self.insur_type}']

    def rank_pred(self, data):
        '''ranking by the mean of k-fold prediction and create the new column of rank result 
            and drop the prediction of k-fold model
    
        Args : 
        -------
            data : dataframe

        Returns : 
        -------
            data : dataframe
        '''
        data[f'rank_{self.insur_type}'] = data[f'mean_{self.insur_type}'].rank(ascending=False)
        data.drop([i for i in data.columns if i.startswith('fold')], axis=1, inplace=True)
        return data

    def save_prediction(self, data, pred_data):
        '''combine the prediction of all insurance type and rename columns
    
        Args : 
        -------
            data : dataframe
                the prediction of data
            pred_data : dataframe
                null dataframe

        Returns : 
        -------
            pred_data : dataframe
                the prediction of all insurance type 
        '''
        if len(pred_data) == 0:
            pred_data = data
        else:
            pred_data = pred_data.merge(data, on=self.conf.cust_id, how='left')

        pred_data = pred_data.rename(columns={f'mean_{self.insur_type}':f'{self.insur_type}_MKT_PROB',
                                              f'rank_{self.insur_type}':f'{self.insur_type}_MKT_RANK'})
        return pred_data

    def predict(self, pred_data, mode):
        '''switch two mode to use this function, in order to predict data in different situation.
    
        Args : 
        -------
            pred_data : dataframe
                null dataframe for storing all prediction of all insurane type
        Returns : 
        -------
            df_pred_alltype : dataframe
                all prediction of all insurane type
        '''
        if mode == 'predict':
            model, features = self.load_model(self.dict_path["path_output"],
                                              f'{self.insur_type}_模型_{self.conf.season_date}',
                                              self.conf.fold_num)
            df_pred = predict_result(self.data, features, model,
                                     fold_num=self.conf.fold_num,
                                     insur_type=self.insur_type)
            self.rank_pred(df_pred)
            df_pred_alltype = self.save_prediction(df_pred, pred_data)
            del df_pred
            gc.collect()
        if mode == 'backtest':
            self.retain_y()
            model, features = self.load_model(self.dict_path["path_output"],
                                              f'{self.insur_type}_模型_{self.conf.season_date}',
                                              self.conf.fold_num)
            df_pred = predict_result(self.data, features, model,
                                     fold_num=self.conf.fold_num,
                                     insur_type=self.insur_type)
            self.paste_y(df_pred)
            self.rank_pred(df_pred)
            df_pred_alltype = self.save_prediction(df_pred, pred_data)
            del df_pred
            gc.collect()
        return df_pred_alltype
        