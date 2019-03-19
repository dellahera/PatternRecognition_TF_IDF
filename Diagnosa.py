import io
import re
import math
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover() 

df =  pd.read_csv('pengpol.csv')
df=df.drop(['No'], axis = 1)
#data yang masih perlu ditentukan diagnosanya
data= df[df.isnull().any(axis=1)]
data
df = df.fillna(' ')

#PreProcessing
filter_G=[]
filter_D=[]
for i in range(1, 100):
    #casefolding
    df['Gejala'][i]= df['Gejala'][i].casefold()
    df['Diagnosis'][i]= df['Diagnosis'][i].casefold()
    #filtering
    stop_G=stopword.remove(df['Gejala'][i])
    stop_D=stopword.remove(df['Diagnosis'][i])
    #tokenisasi
    replacements = (",", "-", "!", "?", "\n", "(", ")", ":")
    for r in replacements:
        stop_G= stop_G.replace(r, ' ')
        stop_D= stop_D.replace(r, ' ')
    filter_G.append(stop_G.split())
    filter_D.append(stop_D.split())
	
gejala= filter_G
diagnosa= filter_D

#Term Frequency
count_word={}
def TF(data):
    tf_word={}
    for i in data:
        if i in tf_word:
            tf_word[i]+1
        else:
            tf_word[i]=1
        if i in count_word:
            count_word[i] +=1
        else:
            count_word[i] =1
    for i in tf_word:
        tf_word[i] = tf_word[i] / len(i)
    return tf_word
	
data_TF=[]
for i in range(len(gejala)):
    tf = TF(gejala[i])
    data_TF.append(tf)

def IDF():
    idfDict = {}
    for i in count_word:
        idfDict[i] = math.log(len(data) / count_word[i])
    return idfDict
	
idf = IDF()

def TFIDFDict(data):
    TFIDF_ = {}
    for i in data:
        TFIDF_[i] = data[i] * idf[i]
    return TFIDF_

tfidf = [TFIDFDict(review) for review in data_TF]

worddict = sorted(count_word.keys())

def vektorTFIDF(text):
    vektortfidf = len(worddict) * [0.0]
    for i, word in enumerate(worddict):  
        if word in text:
            vektortfidf[i] = text[word]
    return vektortfidf

vektortfidf = [vektorTFIDF(x) for x in tfidf]
def magmagnitude(vector):
    return math.sqrt(sum(i**2 for i in vector))

def dot_product(v1,v2):
    return np.dot(v1,v2)/(magnitude(v1)*magmagnitude(v2))

tfidfVector = [TFIDFVector(dat) for dat in tfidf]
cosine_similarity = dot_product(tfidfVector[0], tfidfVector[1]) / magnitude(tfidfVector[0]) * magnitude(tfidfVector[1])
for x in data.index:
    similarity = []
    for i in range(len(vektortfidf)):
        if (i !=x and i not in data.index):
            co = dot_product(vektortfidf[i], vektortfidf[x])
            similarity.append(co)
    maxx = similarity.index(max(similarity))
    data['Diagnosis'][x] = df['Diagnosis'][maxx]
    print("\nIndex ", x, "\nData gejala : ", data["Gejala"][x], " \nDiagnosis : ",data['Diagnosis'][x], "\nKemiripan : ",max(similarity))