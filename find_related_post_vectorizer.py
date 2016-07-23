from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
from  nltk.stem.snowball import SnowballStemmer
'''
content = ["How to format my hard disk", "Hard disk format problems "]
vectorizer=CountVectorizer(min_df=1)
token=vectorizer.fit_transform(content)
print token
print vectorizer.get_feature_names()
print token.toarray().T
'''

'''
directory read

'''
stemer=SnowballStemmer("english")
class SnowballCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer=super(SnowballCountVectorizer,self).build_analyzer()
        return lambda doc:(stemer.stem(w) for w in analyzer(doc))
DIR="data/toy"
import os,sys
posts=[open(os.path.join(DIR,f)).read() for f in os.listdir(DIR)]
#vec=CountVectorizer(min_df=1,stop_words="english")
vec=SnowballCountVectorizer(min_df=1,stop_words="english")
X_train=vec.fit_transform(posts)
num_samples,num_features=X_train.shape
#print X_train.shape
#print vec.get_feature_names()
#print X_train.toarray()
#new_post="imaging databases"
#new_post_vect=vec.transform([new_post])
#print new_post_vect
#print  new_post_vect.toarray()
'''
(5, 25)
[u'about', u'actually', u'capabilities', u'contains', u'data', u'databases', u'images', u'imaging', u'interesting', u'is', u'it', u'learning', u'machine', u'most', u'much', u'not', u'permanently', u'post', u'provide', u'save', u'storage', u'store', u'stuff', u'this', u'toy']
[[1 1 0 1 0 0 0 0 1 1 1 1 1 0 1 1 0 1 0 0 0 0 1 1 1]
 [0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0]
 [0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0]
 [0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 3 3 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0]]
  (0, 5)	1
  (0, 7)	1
[[0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

'''
def dist_raw(v1,v2):
    v1=v1/sp.linalg.norm(v1.toarray())
    v2=v2/sp.linalg.norm(v2.toarray())
    delta=v1-v2
    return sp.linalg.norm(delta.toarray())
new_post="imaging databases"
new_post_vec=vec.transform([new_post])
best_i=None
best_doc=None
best_dist=sys.maxint
for i in range(num_samples):
    post=posts[i]
    if post==new_post:
        continue
    post_vect=X_train.getrow(i)
    #print post_vect.toarray()
    d=dist_raw(post_vect,new_post_vec)
    print "=== dist post :%i with dist ==%.2f" % (i, d)
    if d<best_dist:
        best_dist=d
        best_i=i
print "===best dist post :%i with best i ==%.2f"%(best_i,best_dist)
