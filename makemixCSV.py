#encoding=utf-8
import numpy as np
import pandas as pd
#data1 = pd.read_csv('./data/white_mix.txt',delimiter='\t', encoding='utf-8',names=['uri','label'])
data1 = pd.read_csv('./data/white_mix.txt', encoding='utf-8',names=['uri'])
#data1.to_csv('./data/white_mix.csv')

#data2 = pd.read_csv('./data/black_mix.txt',delimiter='\t', encoding='utf-8',names=['uri','label'])
data2 = pd.read_csv('./data/black_mix.txt', encoding='utf-8',names=['uri'])
#data2.to_csv('./data/black_mix.csv')
data = pd.concat([data1,data2])

data.to_csv('./data/mix.csv', index=False,header=True)