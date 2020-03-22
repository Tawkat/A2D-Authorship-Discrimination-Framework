import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df_Quran=pd.read_csv('Quran.csv')
df_Quran=df_Quran.iloc[:,1:]



for i in range(len(df_Quran['quotes'])):
    if(len(df_Quran['quotes'][i].strip())==0):
        df_Quran.at[i,'quotes']=''

df_Quran['quotes'].replace('',np.nan,inplace=True)
isnull=df_Quran['quotes'].isnull().sum()


df_Quran.dropna(how='any',inplace=True)


df_Bukhari=pd.read_csv('Bukhari.csv')
df_Bukhari=df_Bukhari.iloc[:,1:]



for i in range(len(df_Bukhari['quotes'])):
    if(len(df_Bukhari['quotes'][i].strip())==0):
        df_Bukhari.at[i,'quotes']=''

df_Bukhari['quotes'].replace('',np.nan,inplace=True)
isnull=df_Bukhari['quotes'].isnull().sum()


df_Bukhari.dropna(how='any',inplace=True)



df=pd.concat([df_Quran,df_Bukhari],ignore_index=True)
df=pd.DataFrame(df)



df_shuffle=shuffle(df)
df_shuffle=shuffle(df_shuffle)

df_shuffle=pd.DataFrame(df_shuffle)
df_shuffle.to_csv('Quran_Bukhari.csv',index=False)

