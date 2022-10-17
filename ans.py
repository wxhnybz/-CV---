import pandas as pd
from predict import *
df_submit = pd.read_csv('/Users/wangzhiyi/Desktop/CV/预测结果/mchar_sample_submit_A.csv')
df_submit['file_code'] = test_predict
df_submit.to_csv('submit.csv', index=None)