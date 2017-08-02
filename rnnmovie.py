#http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#LSTM

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 50#A comment may not have more than 50 words
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()

#Embedding converts positive indices to dense vectors of equal length
#Represents each word(totally 500) as vectors of size 32
#Usually first layer
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))


model.add(SimpleRNN(100,input_shape=(None,50,32)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train,epochs=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Accuracy : 74.49%