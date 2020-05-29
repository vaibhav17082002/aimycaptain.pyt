import sys
import scipy
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import sklearn.metrics


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB


url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names=['sepal length','sepal with','petal lenght','petal with','class']
dataset= read_csv(url,names=names)


print(dataset.shape)


print(dataset.head(20))

print(dataset.describe())

print(dataset.groupby('class').size())

dataset.plot(kind='box',subplots=True,layout=(2,2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()


scatter_matrix(dataset)
pyplot.show()


array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train,x_valuation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=1)



models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB())) 
models.append(('SVM',SVC(gamma='auto'))) 




results=[]
names=[]
for name,model in models:
    kfold =StratifiedKFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' %(names,cv_results.mean(),cv_results.std()))
    
    
    
    
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparision')
pyplot.show()



model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model.predict(x_valuation)



print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))


