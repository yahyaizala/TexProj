from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
import scipy as sp
from  nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
class STfidVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        snw=SnowballStemmer("english")
        an=super(STfidVectorizer,self).build_analyzer()
        return lambda doc:(snw.stem(w) for w in an(doc))
vectorizer=STfidVectorizer(min_df=10,max_df=0.5,decode_error="ignore",stop_words="english")
groups = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'comp.windows.x', 'sci.space']
features=fetch_20newsgroups(subset="train",categories=groups)
vectorized=vectorizer.fit_transform(features.data)
num_samples,num_features=vectorized.shape
n_clusters=50
km=KMeans(n_clusters=50,init="random",verbose=1,random_state=3,n_init=1)
km.fit(vectorized)
new_post="Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks."
new_post_vec=vectorizer.transform([new_post])
label=km.predict(new_post_vec)[0]
smiliar_indices=(km.labels_==label).nonzero()[0]
smilar=[]
for i in smiliar_indices:
    dist=sp.linalg.norm((new_post_vec-vectorized[i]).toarray())
    smilar.append((dist,features.data[i]))
smilar=sorted(smilar)
print smilar[:3]
print len(smilar)
'''
[(1.0378441731334072,
 u"From: Thomas Dachsel <GERTHD@mvs.sas.com>\nSubject: BOOT PROBLEM with IDE controller\nNntp-Posting-Host:
  sdcmvs.mvs.sas.com\nOrganization: SAS Institute Inc.\nLines: 25\n\nHi,\nI've got a Multi I/O card (IDE controller
  + serial/parallel\ninterface) and two floppy drives (5 1/4, 3 1/2) and a\nQuantum
  ProDrive 80AT connected to it.\nI was able to format the hard disk, but I could not boot from\nit.
  I can boot from drive A: (which disk drive does not matter)\nbut if
   I remove the disk from drive A and press the reset switch,\nthe LED of drive A:
   continues to glow, and the hard disk is\nnot accessed at all.\nI guess this must be a problem of either the Multi I/o
   card\nor floppy disk drive settings (jumper configuration?)\nDoes someone have any hint what could be the reason
   for it.\nPlease reply by email to GERTHD@MVS.SAS.COM\nThanks,\nThomas
   \n+-------------------------------------------------------------------+\n| Thomas Dachsel
                                                   |\n|
                                                    Internet: GERTHD@MVS.SAS.COM
    |\n| Fidonet:  Thomas_Dachsel@camel.fido.de (2:247/40)
|\n| Subnet:   dachsel@rnivh.rni.sub.org (UUCP in Germany, now active) |\n| Phone:
 +49 6221 4150 (work), +49 6203 12274 (home)             |\n| Fax:      +49 6221 415101
|\n| Snail:    SAS Institute GmbH, P.O.Box 105307, D-W-6900 Heidelberg |\n| Tagline:
 One bad sector can ruin a whole day...
 |\n+-------------------------------------------------------------------+\n"),
 (1.0494693076510364, u"From: rogntorb@idt.unit.no (Torbj|rn Rognes)\nSubject: Adding int. hard disk drive to IIcx\nKeywords: Mac IIcx, internal, hard disk drive, SCSI\nReply-To: rogntorb@idt.unit.no (Torbj|rn Rognes)\nOrganization: Div. of CS & Telematics, Norwegian Institute of Technology\nLines: 32\n\nI haven't seen much info about how to add an extra internal disk to a\nmac. We would like to try it, and I wonder if someone had some good\nadvice.\n\nWe have a Mac IIcx with the original internal Quantum 40MB hard disk,\nand an unusable floppy drive. We also have a new spare Connor 40MB\ndisk which we would like to use. The idea is to replace the broken\nfloppy drive with the new hard disk, but there seems to be some\nproblems:\n\nThe internal SCSI cable and power cable inside the cx has only\nconnectors for one single hard disk drive.\n\nIf I made a ribbon cable and a power cable with three connectors each\n(1 for motherboard, 1 for each of the 2 disks), would it work?\n\nIs the IIcx able to supply the extra power to the extra disk?\n\nWhat about terminators? I suppose that i should remove the resistor\npacks from the disk that is closest to the motherboard, but leave them\ninstalled in the other disk.\n\nThe SCSI ID jumpers should also be changed so that the new disk gets\nID #1. The old one should have ID #0.\n\nIt is no problem for us to remove the floppy drive, as we have an\nexternal floppy that we can use if it won't boot of the hard disk.\n\nThank you!\n\n----------------------------------------------------------------------\nTorbj|rn Rognes                            Email: rogntorb@idt.unit.no\n"),
 (1.1063375279728889, u'From: im14u2c@camelot.bradley.edu (Joe Zbiciak)\nSubject: Re: Booting from B drive\nNntp-Posting-Host: camelot.bradley.edu\nOrganization: Happy Campers USA\nLines: 25\n\nIn <C5nvvx.ns@mts.mivj.ca.us> rpao@mts.mivj.ca.us (Roger C. Pao) writes:\n[much discussion about switching 5.25" and 3.5" drives removed]\n\nAnother (albeit strange) option is using a program like 800 II\n(available via anonymous FTP at many major sites), or FDFORMAT\n(also available via anonymous FTP), that allows you to format\n5.25HD disks to 1.44Meg, or 3.5"HD disks to 1.2Meg (along with\nmany MANY other formats!) so you can DISKCOPY (yes, the broken\nMeSsy-DOS DISKCOPY!) the 5.25" disks onto 3.5" disks or vice\nversa...  I use this techniques with "NON-DOS" self-booting \ngame disks on my old Tandy 1000, and it works...  Another program\nnamed Teledisk (shareware--available on many major BBS\'s) will\nalso make the weird format disks, provided you have 800 II\nor FDFormat installed....  Some disks that won\'t DISKCOPY\nproperly can be readily Teledisk\'d into the proper format...\nAt least this is a software solution for a hardware/BIOS \ndeficiency, eh? \n\n\n--\nJoseph Zbiciak                         im14u2c@camelot.bradley.edu\n[====Disclaimer--If you believe any of this, check your head!====]\n------------------------------------------------------------------\n\nNuke the Whales!\n')]
166


'''