import os,collections,csv
import numpy as np

DATA_DIR="data"
FILE_NAME="SentiWordNet_3.0.0_20130122.txt"
def load_wordnet():
    score=collections.defaultdict(list)
    with open(os.path.join(DATA_DIR,FILE_NAME)) as csvFile:
        reader=csv.reader(csvFile,delimiter='\t',quotechar='"')
        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line)<2:
                continue
            POS,ID,POSSCORE,NEGSCORE,SYNSETTERMS,GLOSS=line
            for term in SYNSETTERMS.split(" "):
                term=term.split("#")[0]
                term=term.replace("-"," ").replace("_"," ")
                key="%s/%s"%(POS,term.split("#")[0])
                try:
                    score[key].append((float(POSSCORE),float(NEGSCORE)))
                except:
                    continue
    for key,val in score.items():
        if np.mean(val,axis=0) is None:
            continue
        score[key]=np.mean(val,axis=0)
        #print score[key],"------",key,"------",val
        '''
[ 0.  0.] ------ n/hoagland ------ [(0.0, 0.0)]
[ 0.04166667  0.        ] ------ v/inspect ------ [(0.0, 0.0), (0.0, 0.0), (0.125, 0.0)]
[ 0.125  0.   ] ------ n/matinee ------ [(0.125, 0.0)]
[ 0.  0.] ------ n/joseph lincoln steffens ------ [(0.0, 0.0)]
[ 0.  0.] ------ n/dative ------ [(0.0, 0.0)]
[ 0.  0.] ------ v/tailor make ------ [(0.0, 0.0), (0.0, 0.0)]
[ 0.  0.] ------ n/ocean state ------ [(0.0, 0.0)]
[ 0.125  0.   ] ------ n/anseres ------ [(0.125, 0.0)]
[ 0.  0.] ------ n/saphar ------ [(0.0, 0.0)]
[ 0.     0.125] ------ v/walk off ------ [(0.0, 0.0), (0.0, 0.25)]
[ 0.  0.] ------ n/pearmain ------ [(0.0, 0.0)]
[ 0.  0.] ------ n/clamp ------ [(0.0, 0.0)]
[ 0.  0.] ------ n/clams ------ [(0.0, 0.0)]
[ 0.  0.] ------ n/osip emilevich mandelstam ------ [(0.0, 0.0)]
[ 0.  0.] ------ n/cottier ------ [(0.0, 0.0)]
[ 0.  0.] ------ n/great ------ [(0.0, 0.0)]

        '''
    return score


#print  load_wordnet()