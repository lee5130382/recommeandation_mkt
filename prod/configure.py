'''
configure for testing
'''
class Configure:
    def __init__(self):
        self.pred_data_name = 'MKT_COMB_COLS_20230515.csv' # backtest data file name (SAS ABT時間)
        self.season_date = '2023_04_06' # 時間:定義預測月份/產出時間區間
        self.times = '_3'
        self.chunk_size = 100000 # setting chunk size to load data efficiently
        self.cust_id = 'ID' # 客戶id欄位
        self.target_y = ['HOS', 'SUR', 'REI', 'ACC', 'DD', 'LTC', 'LIFE']
        self.fold_num = 5 # fold數量設定
        self.anal_rate = [0.1, 0.2, 0.3] # 分檻
