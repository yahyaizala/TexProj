import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
data=sp.genfromtxt("data/web_traffic.tsv",delimiter="\t")
x=data[:,0]
y=data[:,1]
x=x[~sp.isnan(y)]
y=y[~sp.isnan(y)]
plt.scatter(x,y,s=10)
plt.title("Web Traffic over last moth")
plt.xticks([w*7*24 for w in range(10)],["Week%i"%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True,linestyle="-",color="0.75")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
fp1,residuals,rank,sv,rcond=sp.polyfit(x,y,53,full=True)
def error(f,x,y):
    return sp.sum((f(x)-y)**2)
f1=sp.poly1d(fp1)
print error(f1,x,y)
fx=sp.linspace(0,x[-1],1000)
print fx
plt.plot(fx,f1(fx),linewidth=4)
print x[-1]
plt.legend(["d=%i"%f1.order],loc="upper left")
fbx=sp.poly1d(sp.polyfit(x,y,2))
reach=fsolve(fbx-10000,x0=800)/(7*24)
print reach
plt.show()

