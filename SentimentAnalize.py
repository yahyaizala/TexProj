import os,nltk
import json
DATA_DIR="data\\tweet"
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC,SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import precision_recall_curve,auc
from sklearn.base import BaseEstimator
import re
def load_sanders_data(dirname="", line_count=-1):
    count = 0
    topics = []
    labels = []
    tweets = []

    with open(os.path.join(DATA_DIR, dirname, "corpus.csv"), "r") as csvfile:
        metareader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in metareader:
            count += 1
            if line_count > 0 and count > line_count:
                break

            topic, label, tweet_id = line

            tweet_fn = os.path.join(
                DATA_DIR, dirname, 'rawdata', '%s.json' % tweet_id)
            try:
                tweet = json.load(open(tweet_fn, "r"))
            except IOError:
                print("Tweet '%s' not found. Skip."%tweet_fn)
                continue

            if 'text' in tweet and tweet['user']['lang'] == "en":
                topics.append(topic)
                labels.append(label)
                tweets.append(tweet['text'])

    tweets = np.asarray(tweets)
    labels = np.asarray(labels)

    return tweets, labels

def n_gram():
    clf=MultinomialNB(alpha=0.05,fit_prior=True)
    #clf=BernoulliNB()
    #clf=LinearSVC()
    #clf=SVC()
    tfidf = TfidfVectorizer(ngram_range=(1, 3),binary=False,
                            analyzer="word",sublinear_tf=True,smooth_idf=True,
                            min_df=1,max_df=0.5,stop_words="english")
    pipe=Pipeline([("vect",tfidf),("clf",clf)])
    return pipe
def train_model(clf,X,Y):
    cv=ShuffleSplit(n=len(X),n_iter=10,test_size=0.3,random_state=0)
    scores=[]
    pr_scores=[]
    clf=clf
    for train,test in cv:
        X_train,y_train=X[train],Y[train]
        X_test,y_test=X[test],Y[test]
        clf.fit_transform(X_train,y_train)
        train_score=clf.score(X_train,y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)
        proba=clf.predict_proba(X_test)
        precision,recal,thresh=precision_recall_curve(y_test,proba[:,1])
        pr_scores.append(auc(recal,precision))
        summary=(np.mean(scores),np.std(scores),np.mean(pr_scores),np.std(pr_scores))
        print "%.3f\t %.3f \t %.3f \t %.3f"%summary
def tweak_labels(Y,pos_list):
    pos=Y==pos_list[0]
    for item in pos_list[1:]:
        pos |=Y==item
    Y=np.zeros(Y.shape[0])
    Y[pos]=1
    Y=Y.astype(int)
    print "==Y",Y
    return Y
def grid_search(clf,X,Y):
    cv=ShuffleSplit(n=len(X),n_iter=10,test_size=0.3,random_state=0)
    grid=dict(vect__ngram_range=[(1,1),(1,2),(1,3)],
              vect__min_df=[1,2],
              vect__stop_words=[None,"english"],
              vect__use_idf=[False,True],
              vect__sublinear_tf=[False,True],
              vect__binary=[False,True],
              clf__alpha=[0.1,0.05,0.01,0.5,1])
    search=GridSearchCV(clf,param_grid=grid,
                        cv=cv,
                        score_func=f1_score,verbose=10)
    search.fit(X,Y)
    return search.best_estimator_

X,Y=load_sanders_data()
pos_neg_idx=np.logical_or(Y=="positive",Y=="negative")
X=X[pos_neg_idx]
Y=Y[pos_neg_idx]
Y=Y=="positive"
#Y=tweak_labels(Y,["positive","negative"])
#train_model(n_gram(),X,Y)
#clf=grid_search(n_gram(),X,Y)
#print clf
'''WORDNETYardim'''
from Pos_Tagger_SentiWordNet import load_wordnet
sent_word_net=load_wordnet()
class LinguisticVectorizer(BaseEstimator):
    def get_feature_names(self):
        return np.array(["sent_neut","sent_pos","sent_neg","nouns","adjectives","verbs","advverbs",
                         "allcaps","exclamation","quetion","hashtag","mentioning"])
    def fit(self,documents,y=None):
        return self
    def _get_sentiments(self,d):
        sent=tuple(nltk.word_tokenize(d))
        tagged=nltk.pos_tag(sent)
        pos_vals=[];neg_vals=[];nouns=0;adjectives=0;verbs=0;adverbs=0;
        for w,t in tagged:
            p,n=0,0
            sent_pos_type=None
            if t.startswith("NN"):
                sent_pos_type="n"
                nouns +=1
            elif t.startswith("JJ"):
                sent_pos_type="a"
                adjectives +=1
            elif t.startswith("VB"):
                sent_pos_type="v"
                verbs +=1
            elif t.startswith("RB"):
                sent_pos_type="r"
                adverbs +=1
            if sent_pos_type is not None:
                sent_word="%s/%s"%(sent_pos_type,w)
                if sent_word in sent_word_net:
                    p,n=sent_word_net[sent_word]
                    if p and n:
                        pos_vals.append(p)
                        neg_vals.append(n)
        l=len(sent)
        av_pos_val=np.mean(pos_vals)
        av_neg_val=np.mean(neg_vals)
        return [1-(av_pos_val+av_neg_val),av_pos_val,av_neg_val,nouns/l,adjectives/l,verbs/l,adverbs/l]
    def transform(self,documents):
        obj_val,pos_vals, neg_vals,nouns,adjectives,verbs,adverbs =np.array([self._get_sentiments(d)
                                                                            for d in documents]).T
        allcaps = []
        question = []
        hashtag = []
        mentioning =[]
        exclamation=[]
        for d in documents:
            allcaps.append(np.sum([t.isupper() for t in d.split() if len(t)>2]))
            exclamation.append(d.count("!"))
            question.append(d.count("?"))
            hashtag.append(d.count("#"))
            mentioning.append(d.count("@"))
        res=np.array([obj_val,pos_vals,neg_vals,nouns,verbs,adverbs,allcaps,exclamation,question,hashtag,mentioning]).T
        return res

emo_repl = {
    # positive emoticons
    "&lt;3": " good ",
    ":d": " good ",  # :D in lower case
    ":dd": " good ",  # :DD in lower case
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    # negative emoticons:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
}

emo_repl_order = [k for (k_len, k) in reversed(
    sorted([(len(k), k) for k in emo_repl.keys()]))]

re_repl = {
    u"\br\b": "are",
    u"\bu\b": "you",
    u"\bhaha\b": "ha",
    u"\bhahaha\b": "ha",
    u"\bdon't\b": "do not",
    u"\bdoesn't\b": "does not",
    u"\bdidn't\b": "did not",
    u"\bhasn't\b": "has not",
    u"\bhaven't\b": "have not",
    u"\bhadn't\b": "had not",
    u"\bwon't\b": "will not",
    u"\bwouldn't\b": "would not",
    u"\bcan't\b": "can not",
    u"\bcannot\b": "can not",
}

def create_union(params=None):
    def preprocessor(tweet):
        tweet=tweet.lower()
        for k in emo_repl_order:
            tweet=tweet.replace(k,emo_repl[k])
        for r,repl in re_repl.items():
            tweet=re.sub(r,repl,tweet)
        return tweet
    tfidfa=TfidfVectorizer(preprocessor=preprocessor,analyzer="word")
    lingua=LinguisticVectorizer()
    all_features=FeatureUnion([("ling",lingua),("tfidf",tfidfa)])
    clf=MultinomialNB()
    pipeLine=Pipeline([("all",all_features),("clf",clf)])
    if params:
        pipeLine.set_params(**params)
    return pipeLine
def get_best_model():
    best_params = dict(all__tfidf__ngram_range=(1, 2),
                       all__tfidf__min_df=1,
                       all__tfidf__stop_words=None,
                       all__tfidf__smooth_idf=False,
                       all__tfidf__use_idf=False,
                       all__tfidf__sublinear_tf=True,
                       all__tfidf__binary=False,
                       clf__alpha=0.01
                       )

    best_clf = create_union(best_params)
    return best_clf


best=create_union()
train_model(best,X,Y)









