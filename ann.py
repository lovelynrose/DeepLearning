#Create a simple neural network with 3 layers to implement the AND function

#Create Sequential Model - a linear stack of layers
from keras.models import Sequential

#Call the constructor
model=Sequential()

#Stack layers
#'Dense' to create fully connected network
#Activation - default linear activation f(x)=x

#Input layer has 2 neurons
from keras.layers import Dense
model.add(Dense(2,input_dim=2))
#compulsorily give input size when creating the first layer
#2-refers to the number of output units
#If user_bias=False - more time is taken to get good accuracy

#hidden layer has 2 neurons
#Without hidden layer more time is taken to get good accuracy
model.add(Dense(2))

#output layer has 1 neuron
model.add(Dense(1,activation='sigmoid'))

#Compile the model
model.compile(optimizer='SGD',loss='mean_squared_error',metrics=['accuracy'])

#Try on simple AND function
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,0,0,1]
#Execute the model on some data
model.fit(X,Y,epochs=1000)
