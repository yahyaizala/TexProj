from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pyplot as plt
import numpy as np
boston=load_boston()
#print list(boston)
#print boston.data[:,5]
#print boston.feature_names
#print boston.target[:5]
'''


['data', 'feature_names', 'DESCR', 'target']
[[  6.32000000e-03   1.80000000e+01   2.31000000e+00   0.00000000e+00
    5.38000000e-01   6.57500000e+00   6.52000000e+01   4.09000000e+00
    1.00000000e+00   2.96000000e+02   1.53000000e+01   3.96900000e+02
    4.98000000e+00]

'''
X=boston.data[:,5]
y=boston.target
X=np.transpose(np.atleast_2d(X))
lr=LinearRegression()
lr.fit(X,y)
mse=mean_squared_error(y,lr.predict(X))
print "Mean {:.3}".format(mse)
r2=r2_score(y,lr.predict(X))
print r2
print lr.score(X,y)



