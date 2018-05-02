#encoding=utf-8
import numpy as np
import pandas as pd
#data1 = pd.read_csv('./data/UGCwhite.txt',delimiter='\t', encoding='utf-8',names=['uri','label'])
data1 = pd.read_csv('./data/UGCwhite.txt', encoding='utf-8',names=['uri'])
data1.to_csv('./data/UGCwhite.csv',index=False,header=True)

#data2 = pd.read_csv('./data/UGCblack.txt',delimiter='\t', encoding='utf-8',names=['uri','label'])
data2 = pd.read_csv('./data/UGCblack.txt', encoding='utf-8',names=['uri'])
data2.to_csv('./data/UGCblack.csv',index=False,header=True)