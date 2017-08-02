#Create Sequential Model - a linear stack of layers
from keras.models import Sequential

#Call the constructor
model=Sequential()
#Stack layers
#'Dense' to create fully connected network
#Activation - default linear activation f(x)=x

#Get input data
from keras.datasets import imdb

#The dataset is preprocessed and words are ordered in decreasing order of frequency of occurrence
#Stores comments as sequences
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=500,#keep the most frequent 500 words
                                                      skip_top=5,#Skip the top words as they may be commonly occuring terms
                                                      maxlen=None,#The maximum length of a sequence allowed
                                                      seed=113,#to shuffle samples
                                                      start_char=1,
                                                     )

#print(x_train)
#print(type(x_train))
print("LENGTH : ")
print(len(x_train))
print(x_train.shape)

#Comments stored as sequence, so use sequence for preprocessing
from keras.preprocessing import sequence,text
max_words=50
tokenizer = text.Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test= tokenizer.sequences_to_matrix(x_test, mode='binary')
#Input layer has as many neurons as num_words
from keras.layers import Dense
#x_padded is of size 25000x2494, so give input_shape as (2494,)
#2494 input neurons to 300 hidden neurons
model.add(Dense(300,input_shape=(50,))) 
#model.add(Dense(300,input_dim=2494))- is the same as the previous line but input_dim is deprecated

#300 hidden units to 300 hidden units
model.add(Dense(300))


#output layer has 1 neuron
#300 hidden neurons to 1 output neuron
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

#Compile the model
model.compile(optimizer='Adam',loss='mean_squared_error',metrics=['accuracy'])


#Execute the model on imdb movie dataset
model.fit(x_train, y_train,epochs=2)


scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Accuracy : 60%