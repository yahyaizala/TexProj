from gensim import corpora,models
from matplotlib import pyplot as P
corpus=corpora.BleiCorpus("data/ap/ap.dat","data/ap/vocab.txt")
model=models.LdaModel(corpus=corpus,alpha=1
                      ,num_topics=100,id2word=corpus.id2word)
doc=corpus.docbyoffset(0)
print doc
topic=model[doc]
num_topic=[len(model[doc]) for doc in corpus]
P.hist(num_topic)
P.show()