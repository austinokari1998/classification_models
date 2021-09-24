#for the dataset below, choose the appropriate classification model to be used 
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv('processed.cleveland.data')
print (data.dtypes)
data['num']=data['num'].astype(float)
data['thal']=data['num'].astype(float)
data['ca']=data['num'].astype(float)


X=np.array(data.drop(['num'],1))

Y=data['num'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)


number=10
newacc=0
for _ in range(number):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
    model=svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)

    ypred=model.predict(X_test)

    acc=accuracy_score(Y_test,ypred)

    if acc>newacc:
        newacc=acc

        print ('accuracy of our model is ', newacc)
    predictions=model.predict(X_test)

for x in range(len(predictions)):
    print ('the predicted is',predictions[x], ' and the actual is', Y_test[x]) 

