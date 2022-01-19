from re import S
from keras import models
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D , MaxPooling2D
from keras import backend as K
from tensorflow import keras
(x_train,y_train) , (x_test,y_test) = mnist.load_data()

y_train=x_train.reshape(x_train.shape[0] , 28,28,1)
y_test=x_test.reshape(x_test.shape[0] , 28,28,1)
input_shape= (28,28,1)

x_train=x_train.astype("float32")
x_test=x_test.astype("float32")

x_test /=255
x_train /=255

y_train=keras.utils.to_categorical(y_train , 10)

y_test=keras.utils.to_categorical(y_test , 10)

#model
batch_size = 128
epoch = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3) , activation= "relu" , input_shape = input_shape))
model.add(Conv2D( 64, (3,3) ,activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128 , activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10 ,activation ="softmax"))
model.compile(loss = keras.losses.categorical_crossentropy , optimizer=keras.optimizers.Adadelta() , metrics=['accuracy'])

Hist = model.fit(x_train ,
y_train , 
batch_size= batch_size , 
epochs=epoch , 
verbose= 1 , 
validation_split=0.2 ,
validation_data=(x_test, y_test))
model.save("mnist.h5")
model.evaluate(x_test , y_test , verbose=0)


