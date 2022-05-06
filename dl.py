"""
=====================
Evaluating Deep Learning classifiers on Use of API Call Features 
extracted using Static and Dynamic Analysis
=====================
This uses Pytorch's dataloader functionality

Run command: 
$ python use.py 
"""

print(__doc__)

import os, sys
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import Dataset, DataLoader

# Cuda for pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def merge(df1,df2):  
    # merge two dataframes, having the same rows, into one dataframes. i.e., add columns/features
    df1 = df1.drop("classLabel", axis=1)
    df1.sort_index(axis=0,inplace=True) # sort indices in ascending order

    df2.sort_index(axis=0,inplace=True) # sort indices in ascending order

    new_df = pd.concat([df1,df2],axis=1)

    return new_df

class CustomDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]

        # Load data and get label
        X = self.data[index]
        y = self.labels[index]

        return X, y

# count param function that shows the specifications of the model
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')


# RNN Specification:
# one LSTM layer
# one linear (fully connected) layer
# output layer with softmax function
class LSTMnetwork(nn.Module):
    def __init__(self, in_features, hidden_sz=30, p=0.25, out_features=2):  
        super(LSTMnetwork, self).__init__()
        self.hidden_sz = hidden_sz

        self.in_features = in_features

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.in_features, hidden_sz)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_sz, out_features)
      
        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1,1,self.hidden_sz),
                       torch.zeros(1,1,self.hidden_sz))

    def forward(self,x):
        lstm_out, self.hidden = self.lstm(
            x.view(len(x), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(x),-1))
        y_val = F.log_softmax(pred, dim=1)
        return y_val


def classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances):
    #print("Running " + name + " classifier on "+ feature_type + " features...")
    global in_features
    in_features = n_train_features
    #global scores
    

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

    # convert into tensor
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    trainDT = CustomDataset(X_train, y_train)
    testDT = CustomDataset(X_test, y_test)
    trainloader = DataLoader(trainDT, batch_size=500, shuffle=True)
    testloader = DataLoader(testDT, batch_size=500, shuffle=False)
    #global name
    for name in names:
        scores = {}
        scores['test_recall'] = []
        scores['test_precision'] = []
        scores['test_f1'] = []
        scores['fit_time'] = []
        print("running " + name + " classifier")
        train_start_time = time.time()

        # initialize ann model
        torch.manual_seed(32)

      
        model = LSTMnetwork(in_features=in_features)
      
        # define loss function to meaure error
        criterion = nn.CrossEntropyLoss()  
      
        # define optimizer to update weights and biases
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001) 
        #model.parameters
        # Train the model
        # use small epoch for large dataset
        # An epoch is 1 run through all the training data
        # losses = [] # use this array for plotting losses
        for _ in range(epochs):
            # using data_loader 
            for i, (data, labels) in enumerate(trainloader):
                # Forward and get a prediction
                # x is the training data which is X_train
                if name.lower() == "rnn":
                    model.hidden = (torch.zeros(1,1,model.hidden_sz),
                        torch.zeros(1,1,model.hidden_sz))
                if name.lower() == "birnn":
                    model.hidden = (torch.zeros(1*2,1,model.hidden_sz),
                        torch.zeros(1*2,1,model.hidden_sz))
              
                y_pred = model.forward(data)
              
                # compute loss/error by comparing predicted out vs acutal labels
                loss = criterion(y_pred, labels)
                #losses.append(loss)
                
                if (i+1)%10==0:  # print out loss for 30 epochs
                    print(f'epoch {_+1} and loss is: {loss}')


                #Backpropagation
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()


        # Validate the model
        train_end_time = time.time()
        fit_time = train_end_time - train_start_time
        correct = 0
        tp=0; tn=0; fp=0; fn=0
        with torch.no_grad(): 
            for _, (data, labels) in enumerate(testloader):
                y_val = model(data)
                predicted = torch.max(y_val.data,1)[1]
                tp,tn,fp,fn,correct = compute_measures(predicted, labels, tp, tn, fp, fn, correct)
            print(f'We got {correct} correct!')

        compute_scores(tp,tn,fp,fn, fit_time, scores)

        # save the model
        #modelName = "../savedModels/" + name.replace(" ","") + "_" + feature_type + "_cv" + str(cv) + '_dl.pt'
        #torch.save(model.state_dict(),modelName)

        # format output
        convertscores2numpy(scores)
        #print(scores)

        result = ""
        result += name + "," + feature_type +","
        result += str(n_train_features) + ","
        result += str(n_train_instances) +  ","
        result += str(n_test_instances) +  ","
        result += "2010-" + str(train_year) +  ","
        result += str(test_year) +  ","
        result += "%0.5f" % (scores['test_recall'].mean()) + ","
        #result += "%0.5f" % (scores['test_recall'].std() * 2) + ","
        result += "%0.5f" % (scores['test_precision'].mean()) + ","
        #result += "%0.5f" % (scores['test_precision'].std() * 2) + ","
        result += "%0.5f" % (scores['test_f1'].mean()) + ","
        #result += "%0.5f" % (scores['test_f1'].std() * 2) + ","
        result += "%0.5f" % (scores['fit_time'].mean()) + ","
        #result += "%0.5f" % (scores['fit_time'].std() * 2) + ","

        #result += str(n_repeats) +  ","
        result += note[0] + ","
        #result += f'{(time.time() - start_time)/3600:.2f},'
        result += str(datetime.datetime.now()) + '\n'

        print(result)
        f = open(outFile, "a+")
        f.write(result)
        f.close()
    

def compute_measures(predicted, labels, tp, tn, fp, fn, correct):

    print("Predicted:\t" + str(predicted))
    print("Actual:\t\t" + str(labels))
    #print(p2)
    for i, y_pred in enumerate(predicted):
        tp,tn,fp,fn = compute_confusionmatrix(y_pred, labels[i], tp, tn, fp, fn)
        
    correct += (predicted==labels).sum()

    return tp, tn, fp, fn, correct
    
def compute_confusionmatrix(y_pred, y_act,tp, tn, fp, fn):
    #print("predicted: " + str(y_pred))
    #print("actual: " + str(y_act))
    if y_act == 1:
        if y_act == y_pred:
            tp+=1
            #print("true positive")
        else:
            fn+=1
            #print("false negative")
    else: 
        if y_act == y_pred:
            tn+=1
            #print("true negative")
        else:
            fp+=1
            #print("false positive")
    return tp, tn, fp, fn

def compute_scores(tp,tn,fp,fn, fit_time, scores):
    try: 
        recall = tp / (tp+fp)
    except: # handle ZeroDivisionError
        recall = 0
    try:
        precision = tp / (tp+fn)
    except:
        precision = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0

    scores['test_recall'].append(recall)
    scores['test_precision'].append(precision)
    scores['test_f1'].append(f1)
    scores['fit_time'].append(fit_time)

    #return scores

def convertscores2numpy(scores):
    scores['test_recall'] = np.array(scores['test_recall'])
    scores['test_precision'] = np.array(scores['test_precision'])
    scores['test_f1'] = np.array(scores['test_f1'])
    scores['fit_time'] = np.array(scores['fit_time'])

if __name__ == '__main__':
    start_time = time.time()

    ################################## Modify following paraemters #################################
    #note = ["only permission features only"]
    note = ["perm:native:refl features"]

    # additional features
    addNative = True
    addRefl = True
    addPerm = True
    
    base = "../datax/dynamic/csv/class_use/dcuf"

    base2 = "../datax/dynamic/csv/native_use/dnuf"
    base3 = "../datax/dynamic/csv/refl_use/druf"    
    base4 = "../datax/dynamic/csv/perm_use/dpuf"
   
    datasets = ["ben_10", "ben_11", "ben_12", "ben_13", "ben_14", "mal_10", "mal_11", "mal_12", 
                "mal_13", "mal_14", "mal_drebin_10_msoft", "mal_drebin_11_msoft", "mal_drebin_12_msoft", 
            "ben_15", "ben_15_msoft", "mal_15", "ben_16", "ben_16_msoft", "mal_16", 
            "ben_17", "ben_17_msoft", "mal_17", "mal_17_msoft", 
            "ben_18_msoft", "mal_18", "mal_18_msoft",
            "mal_19", "mal_19_msoft", "mal_20"]  # 29 items

    ###################################################################################################
    outFile = '../csv_files/scorex_3.csv'

    epochs = 30

    names = ["rnn"]

    # determine type of features that are being used
    fIdx = base.rfind('/') + 1
    feature_type = base[fIdx:]

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
        #note[0] += ":native"
        for ds in datasets:
            inputFile = base2 + "_" + ds + ".csv.gz"
            print(ds)
            df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
            if 'apkName' in df.columns: 
                df = df.drop('apkName', axis=1)
            native_frames.append(df)

    refl_frames = []
    if addRefl:
        #note[0] += ":reflection"
        for ds in datasets:
            inputFile = base3 + "_" + ds + ".csv.gz"
            print(ds)
            df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
            if 'apkName' in df.columns: 
                df = df.drop('apkName', axis=1)
            refl_frames.append(df)

    perm_frames = []
    if addPerm:
        #note[0] += ":perm"
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

            print("Start index: " + str(sIdx) + " , End Index: " + str(eIdx))
            test = get_data_frame(sIdx, eIdx)
            classify(X_train, y_train, test, train_year, test_year, feature_type, n_train_features, n_train_instances)