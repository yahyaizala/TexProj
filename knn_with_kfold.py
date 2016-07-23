import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
features=iris.data
fname=iris.feature_names
tname=iris.target_names
target=iris.target
labels=tname[target]
means=[]
clf=KNeighborsClassifier(n_neighbors=10)
kfold=KFold(n_folds=5,shuffle=False,n=len(features))
for train,test in kfold:
    clf.fit(features[train],labels[train])
    pred=clf.predict(features[test])
    curmean=np.mean(pred==labels[test])
    means.append(curmean)
print "Mean Accuracy {:.1%}".format(np.mean(means))

'''pipeliner'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
clf=Pipeline([("norm",StandardScaler()),("knn",clf)])
for train,test in kfold:
    clf.fit(features[train],labels[train])
    pred=clf.predict(features[test])
    curmean=np.mean(pred==labels[test])
    means.append(curmean)
print "Mean Accuracy {:.1%}".format(np.mean(means))


