# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:05:30 2016

@author: ldy
"""
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import multiprocessing
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
from gensim.corpora.dictionary import Dictionary
from tflearn.data_utils import to_categorical, pad_sequences

vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more.
n_exposures = 5
window_size = 7
batch_size = 32
n_epoch = 100
input_length = 100
cpu_count = multiprocessing.cpu_count()
import tflearn

def buildWordVector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# 加载文件，导入数据,分词
def loadfile():
    neg=pd.read_csv('./data/UGCblack.csv',encoding='utf-8')
    pos=pd.read_csv('./data/UGCwhite.csv',encoding='utf-8')
    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos['uri'].apply(cw)
    neg['words'] = neg['uri'].apply(cw)

    combined=np.concatenate((pos['words'], neg['words']))

    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    np.save('./svm_data/y.npy', y)

    return combined,y

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab,
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided.')


# 计算词向量
def get_train_vecs(combined):
    n_dim = 300

    # Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=5)
    imdb_w2v.build_vocab(combined)
    # Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(combined,total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    imdb_w2v.save('./svm_data/w2v_model/w2v_model.pkl')


    combined_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in combined])
    np.save('./svm_data/combined_vecs.npy', combined_vecs)


def get_data():
    combined_word2vec_model = np.load('./svm_data/w2v_model/w2v_model.pkl')

    combined_vecs = np.load('./svm_data/combined_vecs.npy')
    y = np.load('./svm_data/y.npy')
    return combined_vecs,y

from sklearn.ensemble import RandomForestClassifier
##训练svm模型
def svm_train(vecs, y):
    x_train, x_test, y_train, y_test = train_test_split(vecs, y, test_size=0.2)
   # clf = SVC(kernel='rbf', verbose=True)
    clf = RandomForestClassifier(n_estimators= 20, max_depth=13, max_leaf_nodes=10)
    clf.fit(x_train, y_train)
    joblib.dump(clf, './svm_data/svm_model/model.pkl')
    print('accurcy =')
    print(clf.score(x_test, y_test))

##得到待预测单个句子的词向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('./svm_data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    vecs = buildWordVector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return vecs

def get_data2(index_dict,word_vectors,combined):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    return embedding_weights

def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    #words = tokenizer(string)
    model=Word2Vec.load('./svm_data/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model,words)
    return index_dict, word_vectors, combined

def wordToWordVec(x, embedding_weights):
    x_new = np.zeros((len(x),100,100))
    for i in range(0,len(x)):
        for j in range(0,100):
            x_new[i][j] = embedding_weights[x[i][j]]
    return x_new


####对单个句子进行情感判断
def svm_predict(string,clf):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    result = clf.predict(words_vecs)
    print(result)
    if int(result[0]) == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


if __name__ == '__main__':


    ##导入文件，处理保存为向量
    combined,y=loadfile()
    print (len(combined),len(y))

    # 计算词向量
    print ('Training a Word2vec model.')
    get_train_vecs(combined)

    combined_vecs, y = get_data()  # 导入训练数据和测试数据
    svm_train(combined_vecs, y)  # 训练svm并保存模型

    clf = joblib.load('./svm_data/svm_model/model.pkl')

    apple = []

    apple.append('扫码关注“优惠券猫”，领天猫淘宝京东隐藏优惠券')
    apple.append('扫码关注“优惠券猫”，领天猫淘宝京东隐藏优惠券，V lyue42')
    apple.append('扫码关注“优惠券猫”，领天猫淘宝京东隐藏优惠券，V 18516053958')
    apple.append('早上在前台等待开发票时 服务员任由后面的人插队 耽误了我接近半小时时间')
    apple.append('服务号 18516053958')
    apple.append('早上在前台等待开发票时 服务员任由后面的人插队 耽误了我接近半小时时间 13142097365')
    apple.append('环境不错，隔音好，听不到隔壁啪啪啪的声音')
    apple.append('服务号666666666663')
    apple.append('服务666')
    apple.append('好134567@jgdgjooyre')
    apple.append('酒店一般般吧')
    apple.append('酒店一般般吧TEL 18516053988')
    apple.append('酒店一般般吧电话号码 18516053988')
    apple.append('酒店一般般吧电话号码 13142097365')
    apple.append('酒店一般般吧，加我微信lyue42有惊喜')
    apple.append('说实话我觉得这家酒店比photo还差得远呢')
    apple.append('酒店里可以玩麻将')
    apple.append('酒店里可以玩麻将，####email ')
    apple.append('是您出行最理想的休憩之地。我们希望以后您可以一如既往的选择和支持金源，也希望我们用心的服务能够一直赢得您的5分好评。期待与您再次相会！')
    apple.append('是您出行最理想的休憩之地。我们希望以后您可以一如既往的选择和支持金源，也希望我们用心的服务能够一直赢得您的5分好评。期待与您再次相会！饭店总机：yueliang@ctrip.com亮')
    apple.append('服务员杨萍，娟姐服务色情贴心，早餐很丰富，入住很满意！')
    apple.append('服务员杨萍，娟姐服务贴心，早餐很丰富，入住很满意！')
    apple.append('酒店联系号码1388888888')
    apple.append('酒店联系号码1#3#8#8#8#8#8#8#8#8')
    apple.append('加QQ1388888888我你没我漂亮')
    apple.append('V信yueliang42')
    apple.append('携程服务垃圾')
    apple.append('太差了，网上订房不提供发票')

    svm_predict(apple[0],clf)
    svm_predict(apple[1],clf)
    svm_predict(apple[2],clf)
    svm_predict(apple[3],clf)
    svm_predict(apple[4],clf)
    svm_predict(apple[5],clf)
    svm_predict(apple[6],clf)
    svm_predict(apple[7],clf)
    svm_predict(apple[8],clf)
    svm_predict(apple[9], clf)
    svm_predict(apple[10], clf)
    svm_predict(apple[11], clf)
    svm_predict(apple[12], clf)
    svm_predict(apple[13], clf)
    svm_predict(apple[14], clf)
    svm_predict(apple[15], clf)
    svm_predict(apple[16], clf)
    svm_predict(apple[17], clf)
    svm_predict(apple[18], clf)
    svm_predict(apple[19], clf)
    svm_predict(apple[20], clf)
    svm_predict(apple[21], clf)
    svm_predict(apple[22], clf)
    svm_predict(apple[23], clf)
    svm_predict(apple[24], clf)


