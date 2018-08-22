import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.pyplot import specgram

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_name):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
    	for fn in [parent_dir+'/'+sub_dir+'/'+i+'.wav' for i in file_name]:
            #print fn
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(str(fn))
            except Exception as e:
              #print "Error encountered while parsing file: ", fn
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, sub_dir)
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir_tr = 'data/data_tr'
parent_dir_ts = 'data/data_ts'
tr_sub_dirs = [str(i)for i in range (10)]
ts_sub_dirs = [str(i)for i in range (10)]
file_tr = [str(i) for i in range (1,61)]
file_ts = [str(i) for i in range (61,65)]
tr_features, tr_labels = parse_audio_files(parent_dir_tr,tr_sub_dirs,file_tr)
ts_features, ts_labels = parse_audio_files(parent_dir_ts,ts_sub_dirs,file_ts)
"""import pandas as pd 
f1 = pd.DataFrame(tr_features)
f1.to_csv("tr_features.csv")
f2 = pd.DataFrame(ts_features)
f2.to_csv("ts_features.csv")
f3 = pd.DataFrame(tr_labels)
f3.to_csv("tr_labels.csv")
f4 = pd.DataFrame(ts_labels)
f4.to_csv("ts_labels.csv")
exit()
import pandas as pd
tr_features, tr_labels = pd.read_csv('tr_features.csv').values, pd.read_csv('tr_labels.csv').values
ts_features, ts_labels = pd.read_csv('ts_features.csv').values, pd.read_csv('ts_labels.csv').values
"""

#tr_labels = one_hot_encode(tr_labels)
#ts_labels = one_hot_encode(ts_labels)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, Lasso
try:
    model     = GaussianNB()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'nb: '+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
try:
    model     = RandomForestClassifier()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'rf: '+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
try:
    model     = ExtraTreesClassifier()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'erf:'+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
try:
    model     = LogisticRegression()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'lr: '+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
try:
    model     = LogisticRegressionCV()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'lrcv'+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
try:
    model     = SVC()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'svc:'+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
try:
    model     = KNeighborsClassifier()
    model.fit(tr_features, tr_labels)
    pred = model.predict(ts_features)
    print 'knn:'+str(accuracy_score(pred,ts_labels))
except:
    print 'failed'
exit()

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], 
mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.initialize_all_variables()

cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs): 
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels,1))
    this = [(y_true[i],y_pred[i]) for i in range(len(y_true))]
    print this
#    exit()
    print("Test accuracy: ",round(sess.run(accuracy, 
    	feed_dict={X: ts_features,Y: ts_labels}),3))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print "F-Score:", round(f,3)