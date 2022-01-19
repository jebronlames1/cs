import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle
from imaje_resije import *

dataframe = pd.read_csv('csv/dataset6labels.csv')                  #reads csv file(dataset)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)        # frac is percentage of data u want returned to u (1=100%)
#print(dataframe)                                                  #true for changes to carry over



X = dataframe.drop(['label'], axis=1)                              #dataframe is a matrix, the drop function drops that particular row
Y = dataframe['label']                                             #and axis =1 meaning itll delte a colounm 
                                                                   #seperating image and label from dataset

X_train, Y_train =  X, Y                                           #x=image
X_test,Y_test = X,Y     

print(Y_test[40])
                                                                  #y = label
grid_data = X_train.values[40].reshape(28,28)                     #reshaping matrix to size 28*28
plt.imshow(grid_data,cmap="gray")               
plt.title(Y_train.values[40])                                     #Y_test[40] gives which no it is 
plt.show()


model = svm.SVC(kernel="linear",C=2)
model.fit(X_train,Y_train)
joblib.dump(model, "model/svm_0to5label_linear_2") 
print ("predicting -----------")
predictions = model.predict(X_test)
print ("Accuracy ------------", metrics.accuracy_score(Y_test, predictions))

x =image_resize("6.png")
print(x.shape)
y =model.predict(x)
print(y)
