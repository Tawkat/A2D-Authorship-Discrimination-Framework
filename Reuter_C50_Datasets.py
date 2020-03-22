import os
import numpy as np
import pandas as pd


dic={}
dic['text']=[]
dic['author']=[]

dic_test={}
dic_test['text']=[]
dic_test['author']=[]

label=0

for folder in os.listdir('C50/C50train/'):
    folder_path = 'C50/C50train/'+folder

    for filename in os.listdir(folder_path):
        file=folder_path+'/'+filename
        #print(file)
        with open(file) as f:
            text=f.read()
            dic['text'].append(text)
            dic['author'].append(label)

    label=label+1

label=0

train_author=40

for folder in os.listdir('C50/C50test/'):
    folder_path = 'C50/C50test/'+folder
    Tcnt=1

    for filename in os.listdir(folder_path):
        file=folder_path+'/'+filename
        print(Tcnt,file)

        with open(file) as f:
            text=f.read()
            if Tcnt<=train_author:
                dic['text'].append(text)
                dic['author'].append(label)
            else:
                dic_test['text'].append(text)
                dic_test['author'].append(label)
            Tcnt = Tcnt + 1

    label=label+1



#print(dic['text'][0])

df_train=pd.DataFrame(dic)
df_train.to_csv('Reuter_C50_All_train_primary.csv',columns=['text','author'],index=False)

df_test=pd.DataFrame(dic_test)
df_test.to_csv('Reuter_C50_All_test_primary.csv',columns=['text','author'],index=False)





df_new=pd.DataFrame(df_train)

df_shuffle=shuffle(df_new)
df_shuffle=shuffle(df_shuffle)
#print(df_shuffle.head())




df_shuffle=pd.DataFrame(df_shuffle)
print(df_shuffle.shape)
df_shuffle.to_csv('Reuter_C50_All_train.csv',index=False)




df_new=pd.DataFrame(df_test)

df_shuffle=shuffle(df_new)
df_shuffle=shuffle(df_shuffle)
#print(df_shuffle.head())




df_shuffle=pd.DataFrame(df_shuffle)
print(df_shuffle.shape)
df_shuffle.to_csv('Reuter_C50_All_test.csv',index=False)








