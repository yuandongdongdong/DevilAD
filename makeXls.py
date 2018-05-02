#encoding=utf-8
import numpy as np
import pandas as pd
data1 = pd.read_csv('./data/UGCwhite.txt',delimiter='\t', encoding='utf-8',names=['uri','label'])
data1.to_excel('./data/UGCwhite.xls')

data2 = pd.read_csv('./data/UGCblack.txt',delimiter='\t', encoding='utf-8',names=['uri','label'])
data2.to_excel('./data/UGCblack.xls')
