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
plenth=features[:,2]
print features
print iris
is_setosa=(labels=="setosa")
print  plenth[is_setosa]
max_setosa=plenth[is_setosa].max()
min_non_setosa=plenth[~is_setosa].min()
print  min_non_setosa
print  max_setosa
print features.shape[1]
'''
for t in range(3):
    if t==0:
        c="r"
        marker=">"
    if t==1:
        c="g"
        marker="o"
    if t==2:
        c="b"
        marker="x"
    plt.scatter(features[target==t,0],features[target==t,1],marker=marker,c=c)
plt.show()
'''
'''Starting KNNClassifier'''
