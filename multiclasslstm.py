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



#Data prep
#negative files for testing
tn1 = pd.read_csv('test/testneg1.csv',encoding = 'iso-8859-1')
tn2 = pd.read_csv('test/testneg2.csv',encoding = 'iso-8859-1')
tn3 = pd.read_csv('test/testneg3.csv',encoding = 'iso-8859-1')
tn4 = pd.read_csv('test/testneg4.csv',encoding = 'iso-8859-1')
tn5 = pd.read_csv('test/testneg5.csv',encoding = 'iso-8859-1')
tn6 = pd.read_csv('test/testneg6.csv',encoding = 'iso-8859-1')
tn7 = pd.read_csv('test/testneg7.csv',encoding = 'iso-8859-1')
tn8 = pd.read_csv('test/testneg8.csv',encoding = 'iso-8859-1')
tn9 = pd.read_csv('test/testneg9.csv',encoding = 'iso-8859-1')
tn10 = pd.read_csv('test/testneg10.csv',encoding = 'iso-8859-1')

#pos files for testing
tp1 = pd.read_csv('test/testpos1.csv',encoding = 'iso-8859-1')
tp2 = pd.read_csv('test/testpos2.csv',encoding = 'iso-8859-1')
tp3 = pd.read_csv('test/testpos3.csv',encoding = 'iso-8859-1')
tp4 = pd.read_csv('test/testpos4.csv',encoding = 'iso-8859-1')
tp5 = pd.read_csv('test/testpos5.csv',encoding = 'iso-8859-1')
tp6 = pd.read_csv('test/testpos6.csv',encoding = 'iso-8859-1')
tp7 = pd.read_csv('test/testpos7.csv',encoding = 'iso-8859-1')
tp8 = pd.read_csv('test/testpos8.csv',encoding = 'iso-8859-1')
tp9 = pd.read_csv('test/testpos9.csv',encoding = 'iso-8859-1')
tp10 = pd.read_csv('test/testpos10.csv',encoding = 'iso-8859-1')

#negative files for training
trn1 = pd.read_csv('train/trainneg1.csv',encoding = 'iso-8859-1')
trn2 = pd.read_csv('train/trainneg2.csv',encoding = 'iso-8859-1')
trn3 = pd.read_csv('train/trainneg3.csv',encoding = 'iso-8859-1')
trn4 = pd.read_csv('train/trainneg4.csv',encoding = 'iso-8859-1')
trn5 = pd.read_csv('train/trainneg5.csv',encoding = 'iso-8859-1')
trn6 = pd.read_csv('train/trainneg6.csv',encoding = 'iso-8859-1')
trn7 = pd.read_csv('train/trainneg7.csv',encoding = 'iso-8859-1')
trn8 = pd.read_csv('train/trainneg8.csv',encoding = 'iso-8859-1')
trn9 = pd.read_csv('train/trainneg9.csv',encoding = 'iso-8859-1')
trn10 = pd.read_csv('train/trainneg10.csv',encoding = 'iso-8859-1')

#positive files for training
trp1 = pd.read_csv('train/trainpos1.csv',encoding = 'iso-8859-1')
trp2 = pd.read_csv('train/trainpos2.csv',encoding = 'iso-8859-1')
trp3 = pd.read_csv('train/trainpos3.csv',encoding = 'iso-8859-1')
trp4 = pd.read_csv('train/trainpos4.csv',encoding = 'iso-8859-1')
trp5 = pd.read_csv('train/trainpos5.csv',encoding = 'iso-8859-1')
trp6 = pd.read_csv('train/trainpos6.csv',encoding = 'iso-8859-1')
trp7 = pd.read_csv('train/trainpos7.csv',encoding = 'iso-8859-1')
trp8 = pd.read_csv('train/trainpos8.csv',encoding = 'iso-8859-1')
trp9 = pd.read_csv('train/trainpos9.csv',encoding = 'iso-8859-1')
trp10 = pd.read_csv('train/trainpos10.csv',encoding = 'iso-8859-1')

testfiles = pd.concat([tn1,tn2,tn3,tn4,tn5,tn6,tn7,tn8,tn9,tn10,tp1,tp2,tp3,tp4,tp5,tp6,tp7,tp8,tp9,tp10])
trainfiles = pd.concat([trn1,trn2,trn3,trn4,trn5,trn6,trn7,trn8,trn9,trn10,trp1,trp2,trp3,trp4,trp5,trp6,trp7,trp8,trp9,trp10])

data = pd.concat([testfiles, trainfiles])
#print(data)

#print(data.label.value_counts())


num_of_categories = 1000
shuffled = data.reindex(np.random.permutation(data.index))
e = shuffled[shuffled['label'] == 1][:num_of_categories]
b = shuffled[shuffled['label'] == 2][:num_of_categories]
t = shuffled[shuffled['label'] == 3][:num_of_categories]
m = shuffled[shuffled['label'] == 4][:num_of_categories]
n = shuffled[shuffled['label'] == 5][:num_of_categories]
o = shuffled[shuffled['label'] == 6][:num_of_categories]
p = shuffled[shuffled['label'] == 7][:num_of_categories]
q = shuffled[shuffled['label'] == 8][:num_of_categories]
r = shuffled[shuffled['label'] == 9][:num_of_categories]
s = shuffled[shuffled['label'] == 10][:num_of_categories]
concated = pd.concat([e,b,t,m,n,o,p,q,r,s], ignore_index=True)

#Shuffle the dataset
concated = concated.reindex(np.random.permutation(concated.index))
concated['LABEL'] = 0
#print(concated)

#One-hot encode the lab
concated.loc[concated['label'] == 1, 'LABEL'] = 0
concated.loc[concated['label'] == 2, 'LABEL'] = 1
concated.loc[concated['label'] == 3, 'LABEL'] = 2
concated.loc[concated['label'] == 4, 'LABEL'] = 3
concated.loc[concated['label'] == 5, 'LABEL'] = 4
concated.loc[concated['label'] == 6, 'LABEL'] = 5
concated.loc[concated['label'] == 7, 'LABEL'] = 6
concated.loc[concated['label'] == 8, 'LABEL'] = 7
concated.loc[concated['label'] == 9, 'LABEL'] = 8
concated.loc[concated['label'] == 10, 'LABEL'] = 9

#print(concated['LABEL'][:10])


labels = to_categorical(concated['LABEL'], num_classes=10)
#print(labels[:10])
if 'label' in concated.keys():
    concated.drop(['label'], axis=1)


n_most_common_words = 8000
max_len = 130
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(concated['text'].values)
sequences = tokenizer.texts_to_sequences(concated['text'].values)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)

epochs = 10
emb_dim = 128
batch_size = 256
labels[:2]

#print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.7))
model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])
#print(history.history.keys())

accr = model.evaluate(X_test,y_test, verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


'''
from keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
'''

