"""
Setting the parameter before running main.py
"""
class Configure:
    def __init__(self):
        self.train_data_name = 'TRAIN_ABT_COLS_20221124.csv' # train data file name
        self.backtest_data_name = 'TRAIN_ABT_COLS_20220825.csv' # backtest data file name
        self.season_date = '2023_07_09' # 時間:定義預測月份/產出時間區間
        self.chunk_size = 100000 # setting chunk size to load data efficiently
        self.cust_id = 'ID' # 客戶id欄位
        # 險種 : 住院、手術、實支、意外、重疾、長照、壽險
        self.target_y = ['HOS', 'SUR', 'REI', 'ACC', 'DD', 'LTC', 'LIFE'] 
        self.fold_num = 5 # K-fold flow 的K值設定
        # model parameter
        self.verbose_eval = 200 # 每N行顯示
        self.anal_rate = [0.1, 0.2, 0.3] # 分檻
        self.lgb_params = {
            "num_boost_round": 2000,  # 疊代幾次
            "early_stopping_rounds": 200, 
            "objective": "binary",  # 0101
            "metric": "auc",  # 要看AUC
            "boosting": 'gbdt',  
            "is_unbalance": True,  # 處理imbalance
            "learning_rate": 0.01,  # 每次提升學習率的列表
            "max_depth": 6,  # 分群的最大值
            "num_leaves": 32,  # default:32
            "min_data_in_leaf": 100,  # default:80
            "bagging_freq": 5,
            "bagging_fraction": 0.7,  # 樣本抽樣比例
            "feature_fraction": 0.7,  # 欄位抽樣比例
            "tree_learner": "serial",  # default:serial
            "boost_from_average": "false",
            # "lambda_l1" : 5,
            # "lambda_l2" : 5,
            "bagging_seed": 1020,
            "verbosity": -1,
            "seed": 1020}
        # https://www.cnblogs.com/bjwu/p/9307344.html
        # https://zhuanlan.zhihu.com/p/376485485
