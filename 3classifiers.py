from sklearn import model_selection, preprocessing, linear_model, metrics, svm, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.classify import DecisionTreeClassifier 
from nltk.classify import MaxentClassifier, maxent
from sklearn import decomposition, ensemble
import numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pandas as pd
import math
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
import numpy

numpy.random.seed(7)

# load the dataset
data1 = pd.read_csv('us-economic-newspaper-1.csv',encoding = 'iso-8859-1')
data2 = pd.read_csv('Full-Economic-News-DFE-839861-3.csv',encoding = 'iso-8859-1')
data = pd.concat([data1.loc[data1['relevance']=='yes'].loc[:,['positivity','text']],data2.loc[data2['relevance']=='yes'].loc[:,['positivity','text']]])
#print(data)

texts = data['text']

labels = data['positivity'].fillna(1).values
for i in range(len(labels)):
	if labels[i]<5:
		labels[i] = 1
	elif labels[i]>4:
		labels[i] = 8

print(labels)
		
# create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['positivity'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['positivity'])
#print(train_x, valid_x, train_y, valid_y)

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

def train_model(classifier, feature_vector_train, positivity, feature_vector_valid, is_neural_net=False):
    #fit the training dataset on the classifier
    classifier.fit(feature_vector_train, positivity)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

'''
# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy: ", round(accuracy*100,2),"%")
'''

 
	
# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy Naive Bayes: ", round(accuracy*100,2),"%")
 
accuracy2 = train_model(tree.DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=10), xtrain_count, train_y, xvalid_count)
print("Accuracy Decision tree:",round(accuracy2*100,2),"%")
 
accuracy3 = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("Accuracy logistic Regression:",round(accuracy3*100,2),"%")
 
if accuracy >= accuracy2:
   print("Selected Algorithm is : Naive Bayes with accuracy:",round(accuracy*100,2),"%")
    
elif accuracy2 >= accuracy3:
   print ("Selected Algorithm is : Decision Tree with accuracy:",round(accuracy2*100,2),"%")
   
else:
   print("Selected Algorithm is : Logistic Regression with accuracy:",round (accuracy3*100,2),"%")
 






