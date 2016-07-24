from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
import numpy as np
boston=load_boston()
X=boston.data
y=boston.target
lr=LinearRegression()
lr.fit(X,y)
predicted=lr.predict(X)
'''validation'''
kf=KFold(len(X),n_folds=5)
p=np.zeros_like(y)
for train,test in kf:
    lr.fit(X[train],y[train])
    p[test]=lr.predict(X[test])
rmse_cv=np.sqrt(mean_squared_error(p,y))
print "RMSE of 5-fold cv {:.2}".format(rmse_cv)
'''ElasticNet'''
from sklearn.linear_model import ElasticNetCV
met=ElasticNetCV(n_jobs=-1)
p=np.zeros_like(y)
for t,tst in kf:
    met.fit(X[t],y[t])
    p[tst]=met.predict(X[tst])
p2=r2_score(y,p)
print met.score(X,y)
print p2,"Elastic"





exit()
plt.scatter(predicted,y)
plt.xlabel("Predicted")
plt.ylabel("Actual ")
plt.plot([y.min(),y.max()],[[y.min()],[y.max()]])
plt.show()
