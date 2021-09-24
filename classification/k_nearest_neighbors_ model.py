from sklearn.utils import shuffle 
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
data=pd.read_csv('car.data')
n_obj=preprocessing.LabelEncoder()
buying=n_obj.fit_transform(list(data['buying']))
maint=n_obj.fit_transform(list(data['maint']))
door=n_obj.fit_transform(list(data['door']))
persons=n_obj.fit_transform(list(data['persons']))
lug_boot=n_obj.fit_transform(list(data['lug_boot']))
safety=n_obj.fit_transform(list(data['safety']))
cls=n_obj.fit_transform(list(data['class']))

predict='class'
X=list(zip(buying,maint, door, persons, lug_boot,safety))
Y=list(cls)

X_train,X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1) 
model=KNeighborsClassifier(n_neighbors=9)
model.fit(X_train,Y_train)
acc=model.score(X_test,Y_test)
print ('accuracy',acc)
predictions=model.predict(X_test)
names=['unacc', 'acc', 'good', 'vgood']
for x in range(len(predictions)):
    print ('the prediced ', names[predictions[x]], 'the actual ',names[Y_test[x]])
