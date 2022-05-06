"""
=====================
Evaluating Random Forest Classifier on various features + native + refl + perm features
extracted using static and dynamic analysis
=====================

Run command: 
$ python ml.py 

"""
print(__doc__)

import os, sys
import time
import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
""" from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegressionCV """
from sklearn import preprocessing
#from sklearn.pipeline import make_pipeline
from joblib import dump, load

def merge(df1,df2):  
    # merge two dataframes, having the same rows, into one dataframes. i.e., add columns/features
    df1 = df1.drop("classLabel", axis=1)
    df1.sort_index(axis=0,inplace=True) # sort indices in ascending order

    df2.sort_index(axis=0,inplace=True) # sort indices in ascending order

    new_df = pd.concat([df1,df2],axis=1)

    return new_df

def get_data_frame(idx1, idx2):
    #print("train index: " + str(idx))
    df = pd.concat(frames[idx1:idx2], ignore_index=True)

    if addNative:
        ndf = pd.concat(native_frames[idx1:idx2], ignore_index=True)
        df = merge(df,ndf)
       
    if addRefl:
        rdf = pd.concat(refl_frames[idx1:idx2], ignore_index=True)
        df = merge(df,rdf)

    if addPerm:
        pdf = pd.concat(perm_frames[idx1:idx2], ignore_index=True)
        df = merge(df,pdf)
       
    return df

def classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances):
   
    X_test = test.dropna(axis=1, how='all') # drop columns (axis=1) with 'all' NaN values
    # get data without label
    X_test = X_test.drop('classLabel', axis=1)
    # y = labels
    y_test = test['classLabel']

    n_test_features = len(X_test.columns) # number of features
    n_test_instances = len(X_test) # number of instances
    print("Num test features: " + str(n_test_features))
    print("Num test instances: " + str(n_test_instances))

    # convert to numpy arrays
    X_test = X_test.values
    y_test = y_test.values

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=33)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        start_time = time.time()
        filename = "../savedModels/" + name.replace(" ","") + "_" + feature_type + '_' + str(train_year) + '_' + str(test_year) + '_ml.joblib'
        result = ""
        #note = "no preprocessing"
        note = "native:refl:perm:no preprocessing"
        print("Running " + name + " classifier on "+ feature_type + " features...")
        # this train and test manually
        clf.fit(X_train, y_train)
        #scores = clf.score(X_test, y_test)

        dur = (time.time() - start_time)/3600
        y_pred = clf.predict(X_test)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        #sorted(scores.key())
        dump(clf, filename)


        print("Recall: " + str(recall))
        print("Precision: " + str(precision))
        print("F1: " + str(f1))

        #result = 'Classifier, Type,  Features, Train Instances, Test Instances, Train Year, Test Year, Recall (avg), Recall (std dev), Precision (avg), Precision (std dev), F1 (avg), F1 (std dev), Training Time (avg), Training Time (std dev), Duration, Note, Date\n'
        
        result += name + "," + feature_type +","
        result += str(n_train_features) + ","
        result += str(n_train_instances) +  ","
        result += str(n_test_instances) +  ","
        result += "2010-" + str(train_year) +  ","
        result += str(test_year) +  ","
        result += "%0.5f" % (recall) + ","
        result += "%0.5f" % (precision) + ","
        result += "%0.5f" % (f1) + ","

        result += f'{dur:.2f},'
        result += note + ","
        result += str(datetime.datetime.now()) + '\n'

        f = open(outFile, "a+")
        f.write(result)
        f.close()

################################## Modify following paraemters #################################
base = "../datax/dynamic/csv/class_use/dcuf"
#base = "../datax/dynamic/csv/class_freq/dcfqf"
#base = "../datax/dynamic/csv/class_seq/dcsf"

# additional features
addNative = True
addRefl = True
addPerm = True
base2 = "../datax/dynamic/csv/native_use/dnuf"
#base2 = "../datax/dynamic/csv/native_freq/dnfqf"
#base2 = "../datax/dynamic/csv/native_seq/dnsf"
base3 = "../datax/dynamic/csv/refl_use/druf"    
#base3 = "../datax/dynamic/csv/refl_freq/drfqf"
base4 = "../datax/dynamic/csv/perm_use/dpuf"
#base4 = "../datax/dynamic/csv/perm_freq/dpfqf"
#base4 = "../datax/dynamic/csv/perm_seq/dpsf"

#years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
""" datasets = ["ben_10", "ben_11", "ben_12", "ben_13", "ben_14", "mal_drebin_msoft", "mal_10", "mal_11", "mal_12", 
                "mal_13", "mal_14","ben_15", "mal_15", "ben_16", "mal_16", "ben_17", "mal_17", 
                "ben_msoft", "mal_18", "mal_19", "mal_20", "mal_androzoo_msoft"] """

datasets = ["ben_10", "ben_11", "ben_12", "ben_13", "ben_14", "mal_10", "mal_11", "mal_12", 
                "mal_13", "mal_14", "mal_drebin_10_msoft", "mal_drebin_11_msoft", "mal_drebin_12_msoft", 
            "ben_15", "ben_15_msoft", "mal_15", "ben_16", "ben_16_msoft", "mal_16", 
            "ben_17", "ben_17_msoft", "mal_17", "mal_17_msoft", 
            "ben_18_msoft", "mal_18", "mal_18_msoft",
            "mal_19", "mal_19_msoft", "mal_20"]  # 29 items



###################################################################################################
outFile = '../csv_files/scorex.csv'
names = ["Random Forest"]
classifiers = [
    RandomForestClassifier(n_estimators=100)
]

# determine type of features that are being used
fIdx = base.rfind('/') + 1
feature_type = base[fIdx:]
note = ""
# read all data from csv files
frames = []
for ds in datasets:
    inputFile = base + "_" + ds + ".csv.gz"
    print(ds)
    df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
    if 'apkName' in df.columns: 
        df = df.drop('apkName', axis=1)
    frames.append(df)

native_frames = []
if addNative:
    note += "native:"
    for ds in datasets:
        inputFile = base2 + "_" + ds + ".csv.gz"
        print(ds)
        df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
        if 'apkName' in df.columns: 
            df = df.drop('apkName', axis=1)
        native_frames.append(df)

refl_frames = []
if addRefl:
    note += "reflection:"
    for ds in datasets:
        inputFile = base3 + "_" + ds + ".csv.gz"
        print(ds)
        df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
        if 'apkName' in df.columns: 
            df = df.drop('apkName', axis=1)
        refl_frames.append(df)

perm_frames = []
if addPerm:
    note += "perm:"
    for ds in datasets:
        inputFile = base4 + "_" + ds + ".csv.gz"
        print(ds)
        df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
        if 'apkName' in df.columns: 
            df = df.drop('apkName', axis=1)
        perm_frames.append(df)
    
# ---------------- Dealing with Training dataset --------------------
count = 0
idx = 0
train_year = 2013

for i in range(6):
    train_year += 1
    print("Train Year: " + str(train_year))
    test_year = train_year

    if i == 0: # 2014
        idx = 13
    elif i == 1: # 2015
        idx = 16
    elif i == 2: # 2016
        idx = 19
    elif i == 3: # 2017
        idx = 23
    elif i == 4: # 2018
        idx = 26
    else:   # 2019
        idx = 28
    
    train = get_data_frame(0,idx)

    X_train = train.dropna(axis=1, how='all') # drop columns (axis=1) with 'all' NaN values
    # get data without label
    X_train = X_train.drop('classLabel', axis=1)
    # y = labels
    y_train = train['classLabel']

    n_train_features = len(X_train.columns) # number of features
    n_train_instances = len(X_train) # number of instances
    print("Num train features: " + str(n_train_features))
    print("Num train instances: " + str(n_train_instances))

    # convert to numpy arrays
    X_train = X_train.values
    y_train = y_train.values

    # ---------------- Dealing with Test dataset --------------------
    if i == 0:
        eIdx = idx # 13
        for j in range(6):
            test_year += 1
            print(str(test_year))

            sIdx = eIdx
            if j == 0: # test year 2015
                eIdx = sIdx + 3 # eIdx = 16
            elif j == 1: # test year 2016
                eIdx = sIdx + 3 # eIdx = 19
            elif j == 2: # test year 2017
                eIdx = sIdx + 4 # eIdx = 23
            elif j == 3: # test year 2018
                eIdx = sIdx + 3 # eIdx = 26
            elif j == 4: # test year 2019
                eIdx = sIdx + 2 # eIdx = 28
            else: # test year 2020
                eIdx = sIdx + 1

            # it should be: 13:16, 16:19, 19:23, 23:26, 26:28, 28:29
            print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))
            
            test = get_data_frame(sIdx, eIdx)
            classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)

    elif i == 1:
        eIdx = idx # 16
        for j in range(5):
            test_year += 1
            print(str(test_year))

            sIdx = eIdx # 16
            if j == 0: # test year 2016
                eIdx = sIdx + 3 # sIdx 16, eIdx 19
            elif j == 1: # test year 2017
                eIdx = sIdx + 4 # sIdx 19, eIdx 23
            elif j == 2: # test year 2018
                eIdx = sIdx + 3 # sIdx 23, eIdx 26
            elif j == 3: # test year 2019
                eIdx = sIdx + 2 # sIdx 26, eIdx 28
            else: # test year 2020
                eIdx = sIdx + 1 # sIdx 28, eIdx 29

            # it should be: 16:19, 19:23, 23:26, 26:28, 28:29
            print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))

            test = get_data_frame(sIdx, eIdx)
            classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)

    elif i == 2:
        eIdx = idx  # 19
        for j in range(4):
            test_year += 1
            print(str(test_year))

            sIdx = eIdx # 19
            if j == 0: # test year 2017
                eIdx = sIdx + 4 # sIdx 19, eIdx 23
            elif j == 1: # test year 2018
                eIdx = sIdx + 3 # sIdx 23, eIdx 26
            elif j == 2: # test year 2019
                eIdx = sIdx + 2 # sIdx 26, eIdx 28
            else: # test year 2020
                eIdx = sIdx + 1 # sIdx 28, eIdx 29

            # it should be: 19:23, 23:26, 26:28, 28:29
            print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))
           
            test = get_data_frame(sIdx, eIdx)
            classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)

    elif i == 3:
        eIdx = idx  # 23
        for j in range(3):
            test_year += 1
            print(str(test_year))

            sIdx = eIdx # 23
            if j == 0: # test year 2018
                eIdx = sIdx + 3 # sIdx 23, eIdx 26
            elif j == 1: # test year 2019
                eIdx = sIdx + 2 # sIdx 26, eIdx 28
            else: # test year 2020
                eIdx = sIdx + 1 # sIdx 28, eIdx 29

            # it should be: 23:26, 26:28, 28:29
            print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))

            test = get_data_frame(sIdx, eIdx)
            classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)

    elif i == 4:
        eIdx = idx  # 26
        for j in range(2):
            test_year += 1
            print(str(test_year))

            sIdx = eIdx # 26
            if j == 0: # test year 2019
                eIdx = sIdx + 2 # sIdx 26, eIdx 28
            else: # test year 2020
                eIdx = sIdx + 1 # sIdx 28, eIdx 29

            # it should be: 26:28, 28:29
            print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))

            test = get_data_frame(sIdx, eIdx)
            classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)
    else:
        test_year += 1
        print(str(test_year))
        # test year 2020
        sIdx = idx # 28
        eIdx = sIdx + 1 # sIdx 28, eIdx 29

        # it should be: 28:29
        print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))
        test = get_data_frame(sIdx, eIdx)
        classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)