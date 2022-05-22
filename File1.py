import pandas as pd
import io
import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt


Salarios #DataFrame

x = Salarios['Expe']
X = x[:,np.newaxis]
y= np.array(ASIS['Salario'])


plt.scatter(X,y)
plt.xlabel('Años Exp')
plt.ylabel('Salario')
plt.show()

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
lr = linear_model.LinearRegression()
lr.fit(X_train,y_train)
Y_pred = lr.predict(X_test)


plt.scatter(X_test,y_test)
plt.plot(X_test,Y_pred,color='red',linewidth=3)
plt.xlabel('Años Exp')
plt.ylabel('Salario')
plt.savefig('grafica.png')
plt.show()

import pickle

pickle.dump(lr, open("modelo.pkl", 'wb'))
