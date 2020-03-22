import numpy as np
from nltk import sent_tokenize
import pandas as pd


df={}
df['text']=[]
df['author']=[]

f=open('Rip Sketch/new york.txt',mode='r',encoding='utf-8')

text=f.read()
print(text)
sentences=sent_tokenize(text)

for i in range(len(sentences)):
    sentences[i]=sentences[i].strip()

for i in range(len(sentences)):
    df['text'].append(sentences[i])
    df['author'].append(1)




f4=open('Rip Sketch/The Sketch.txt',mode='r',encoding='utf-8')

text4=f4.read()
print(text4)
sentences4=sent_tokenize(text4)

for i in range(len(sentences4)):
    sentences4[i]=sentences4[i].strip()

for i in range(len(sentences4)):
    df['text'].append(sentences4[i])
    df['author'].append(0)

dataset=pd.DataFrame(df)
dataset.to_csv('New_Sketch.csv',columns=['text','author'],index=False)



from sklearn.utils import shuffle

df=pd.read_csv('New_Sketch.csv')

print(len(df['text']))

df_shuffle=shuffle(df)

df_shuffle = shuffle(df_shuffle)
df_shuffle = shuffle(df_shuffle)
df_ = df_shuffle.tail(1460)
#print(len(df_[df_['author'] == 0]), len(df_[df_['author'] == 1]))

df_shuffle.to_csv('New_Sketch.csv',index=False)

