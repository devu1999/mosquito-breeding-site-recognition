#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2 
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
warnings.filterwarnings("ignore")
from sklearn.externals import joblib


# In[2]:


def getPCA(X,k):
    pca = PCA(n_components=k)
    X_k = pca.fit_transform(X)
    return X_k


# In[3]:


def getLDA(X,k):
    lda = LDA(n_components=k)
    X_k = lda.fit_transform(X)
    return X_k


# In[4]:


def getSift(img):
    img = cv2.imread(img)
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(img,None)


# In[5]:


def getImgRep(img):
    global cnt
    kp,desc=getSift(img)
    return np.mean(np.array(desc).T,axis=1)
    


# In[6]:


dir  = './dip_project/data/'


# In[7]:


def Label_Data():
    ImgData,label=[],[]
    for filename in os.listdir(dir + "rgb_with_puddle"):
        try:
            ret=getImgRep((dir + "rgb_with_puddle/"+filename))
            ImgData.append(ret.ravel())
            label.append(1)
        except:
            pass
    for filename in os.listdir(dir + "rgb_without_puddle"):
        try:
            ret=getImgRep((dir  + "rgb_without_puddle/"+filename))
            ImgData.append(ret.ravel())
            label.append(0)
        except:
            pass

    for filename in os.listdir(dir + "rotated_without_puddle"):
        try:            
            ret=getImgRep((dir  + "rotated_without_puddle/"+filename))
            ImgData.append(ret.ravel())
            label.append(0)
        except:
            pass
    for filename in os.listdir(dir + "rotated_with_puddle"):
        try:
            ret=getImgRep((dir  + "rotated_with_puddle/"+filename))
            ImgData.append(ret.ravel())
            label.append(1)
        except:
            pass
    

    return np.array(ImgData),np.array(label)


# In[8]:


def Split_Data(data,label,x=8):
    X_train,X_test,Y_train,Y_test = train_test_split(np.array(data),np.array(label),test_size=0.2,random_state=x)
    sm = SMOTE()
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
    return X_train,X_test,Y_train,Y_test


# In[9]:


data,label=Label_Data()


# In[10]:


X_train,X_test,Y_train,Y_test=Split_Data(data,label)


# In[28]:


from sklearn.ensemble import VotingClassifier

cv = ShuffleSplit(n_splits=10, test_size=0.2)


models = [
    MultinomialNB(),
    BernoulliNB(),
    LogisticRegression(),
    SGDClassifier(),
    SVC(gamma="auto",kernel='rbf'),
    RandomForestClassifier(),
]

m_names = [m.__class__.__name__ for m in models]

models = list(zip(m_names, models))
vc = VotingClassifier(estimators=models)

sm = SMOTE()


accs = []
f1s = []
cms = []
X_train,X_test,y_train,y_test = train_test_split(np.array(data),np.array(label),test_size=0.2,random_state=86)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

vc.fit(X_train_res, y_train_res)

y_pred = vc.predict(X_test)

accs.append(accuracy_score(y_test, y_pred))
f1s.append(f1_score(y_test, y_pred))
cms.append(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[29]:


print("Voting Classifier")
print("-" * 30)
print("Avg. Accuracy: {:.2f}%".format(sum(accs) / len(accs) * 100))
print("Avg. F1 Score: {:.2f}".format(sum(f1s) / len(f1s) * 100))
print("Confusion Matrix:\n", sum(cms) / len(cms))


# In[34]:


joblib.dump(vc, 'filename_final.pkl') 

