import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model
import nltk
from nltk.tokenize import RegexpTokenizer,word_tokenize, sent_tokenize
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


# Pseudocode
# Ask the user to enter the input text
# The text was tokenized
# Make the tokenize text as list
# Make the prediction based on train model
# 


model = load_model('my_model.h5')


txt = input("Please enter your input: ")

#df4.columns = ['text']
text = sent_tokenize(txt)
#print(text)
#print(len(txt))
txts = [text]
#print(txts)
#txt = ["The effects of an anti-CD3 mAb on induction of non-MHC restricted cytolysis was investigated ."]


for i in txts:
    #i = i.strip()
    #print(i)
    n_most_common_words = 8000
    max_len = 130
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(i)
    sequences = tokenizer.texts_to_sequences(i)
    word_index = tokenizer.word_index
    seq = tokenizer.texts_to_sequences(i)
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    print(pred)


labels = ['Activating invasion and metastasis',
         'Avoiding immune destruction', 
         'Cellular energetics', 
         'Enabling replicative immortality', 
         'Evading growth suppressors', 
         'Genomic instability and mutation', 
         'Inducing angiogenesis', 
         'Resisting cell death', 
         'Sustaining proliferative signaling', 
         'Tumor promoting inflammation']


print('\n')

d={}
def words_freq(s):
    if s in d:
        d[s] +=1
    else:
        d[s] = 1


for i,j in zip(txts[0],pred):
    dicts = {}
    dicts["Sentence"] = i
    dicts["Hallmark"] = labels[np.argmax(j)]
    words_freq(dicts["Hallmark"])
    #print([dicts])
    df1 = pd.DataFrame([dicts])
    #print(df1)

print([d])

#hallmark = dict.keys(d)
sizes = dict.values(d)
#print(hallmark)
#print(sizes)
#def piechart():
#    return hallmark,sizes

total = sum(sizes)
new = [value * 100. / total for value in sizes]
#print (new)

keys = dict.keys(d)

zipobj = zip(keys,new)
dictWords = dict(zipobj)
print(dictWords)


'''
#DRAW PIECHART
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=hallmark, autopct='%1.1f%%',
        shadow=False, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.set_title('Percentage of Cancer Hallmarks\n\n')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
'''




