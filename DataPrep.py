import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import re, SnowballStemmer
from nltk import WordNetLemmatizer

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

    # split into words by white space
    words = text.split()
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in words]
    text = " ".join(text)
    #print(stripped[:100])

    ## Remove puncuation
    #text = text.translate(string.punctuation)

    ########################################################################################
    # replace urls
    re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                        .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                        re.MULTILINE | re.UNICODE)
    # replace ips
    re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

    # replace URLs
    text = re_url.sub("URL", text)

    # replace IPs
    text = re_ip.sub("IPADDRESS", text)
    ####################################################################

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    #stemmer = SnowballStemmer('english')
    #stemmed_words = [stemmer.stem(word) for word in text]
    lemmatizer=WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)


    return text



dataset=pd.read_csv('Reuter_C50/dataset.csv')
texts=[]
texts=dataset['text']
label=dataset['author']

texts=texts.map(lambda x: clean_text(x)) 
print('Total text length: '+str(len(texts)))

X=texts.astype(str).values
X=np.reshape(X,(-1,1))


label=label.astype(int).values
y=np.reshape(label,(-1,1))

import pickle
with open('Reuter_C50/pickle_cleanXy_Reuter_C50_1.pickle','wb') as f:
    pickle.dump((X,y),f)




X_train,y_train,X_test,y_test=texts[:-160],y[:-160,:],texts[-160:],y[-160:,:]

print(len(X_train),len(y_train))

vocabulary_size = 400000
#timeStep=100
timeStep=300
embedding_size=100

##########################################    TRAIN ###################################################

tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
data = pad_sequences(sequences, maxlen=timeStep,padding='post')

print(len(tokenizer.word_index))


vocab_size=len(tokenizer.word_index)+1

f = open('glove.6B.100d.txt',encoding='utf-8')
embeddings={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape[0],embedding_matrix.shape[1])

import pickle
with open('Reuter_C50/pickle_train_Reuter_C50_1_100dim.pickle','wb') as f:
    pickle.dump((data,y_train,embedding_matrix),f)




##########################################    TEST ###################################################



sequences = tokenizer.texts_to_sequences(X_test)
data = pad_sequences(sequences, maxlen=timeStep,padding='post')

print(len(tokenizer.word_index))


import pickle
with open('Reuter_C50/pickle_test_Reuter_C50_1_100dim.pickle','wb') as f:
    pickle.dump((data,y_test,embedding_matrix),f)

