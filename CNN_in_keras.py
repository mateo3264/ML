#from https://keras.io/examples/vision/mnist_convnet/

import keras
from keras.models import Sequential
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import mnist
from sklearn.model_selection import train_test_split
import numpy as np

#Load images
X = mnist.train_images()
y = mnist.train_labels()

#divide dataset into train and test 90% train 10% test
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.1)

#preprocessing
#normalize
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
#add dim
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)
#labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

#build model
def build_cnn(input_shape,n_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),

        Flatten(),
        
        #Dense(512,activation='relu'),
        Dropout(0.5),
        Dense(n_classes,activation='softmax')
        ])

    return model

#init model
model = build_cnn((28,28,1),10)

#define loss,optimizer and metrics
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#train! :D
model.fit(X_train,y_train,epochs=15,batch_size=128,validation_split=0.1,verbose=0)
#test! :D
score = model.evaluate(X_test,y_test,verbose=0)
print('Loss:',score[0])
print('accuracy:',score[1])


