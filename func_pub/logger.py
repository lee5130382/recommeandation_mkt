"""logging工具初始化

初始化logging的工具並供其他script使用,
以方便留存程式執行軌跡.

"""


import os
import sys
import logging

path = os.getcwd()
sys.path.append(path)


logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

# our first handler is a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler_format = '%(asctime)s | %(levelname)s : %(message)s'
console_handler.setFormatter(logging.Formatter(console_handler_format))
logger.addHandler(console_handler)

# the second handler is a file handler
file_handler = logging.FileHandler(os.path.join(path, 'log_of_schedule.log'), encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)
