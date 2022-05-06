# Merge all three types of additional features and perform PCA analysis
import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_blobs
#import smote_variants as sv

def merge(df1,df2):  
    # merge two dataframes, having the same rows, into one dataframes. i.e., add columns/features
    df1 = df1.drop("classLabel", axis=1)
    df1.sort_index(axis=0,inplace=True) # sort indices in ascending order

    df2.sort_index(axis=0,inplace=True) # sort indices in ascending order

    new_df = pd.concat([df1,df2],axis=1)

    return new_df


################ modify base, datasets, outFileName #####################
feature_type = "dcuf_dpuf_druf_dnuf" 
out = "../csv_files/visual_new/"
n_components = 6
dim = n_components # num of PCA

#base = "../datax/dynamic/csv/class_use/dcuf"
#base = "../datax/dynamic/csv/class_freq/dcfqf"
#base = "../datax/dynamic/csv/class_seq/dcsf"
#base = "../datax/static/csv/class_use/scuf"
#base = "../datax/static/csv/class_freq/scfqf"
#base = "../datax/static/csv/class_seq/scsf"

""" bases = ["../datax/dynamic/csv/class_use/dcuf", "../datax/dynamic/csv/class_freq/dcfqf",
            "../datax/dynamic/csv/class_seq/dcsf"
        ] """

""" bases = [ "../datax/static/csv/class_use/scuf", "../datax/static/csv/class_freq/scfqf",
            "../datax/static/csv/class_seq/scsf"
        ]  """


""" datasets = ["ben_10", "ben_11", "ben_12", "ben_13", "ben_14", "mal_10", "mal_11", "mal_12", 
                "mal_13", "mal_14", "mal_drebin_10_msoft", "mal_drebin_11_msoft", "mal_drebin_12_msoft", 
            "ben_15", "ben_15_msoft", "mal_15", "ben_16", "ben_16_msoft", "mal_16", 
            "ben_17", "ben_17_msoft", "mal_17", "mal_17_msoft", 
            "ben_18_msoft", "mal_18", "mal_18_msoft",
            "mal_19", "mal_19_msoft", "mal_20"]  # 29 items """

datasets = ["mal_16", 
           "mal_17", "mal_17_msoft", 
            "mal_18", "mal_18_msoft",
            "mal_19", "mal_19_msoft", "mal_20"]  

datasets = ["ben_10", "ben_11", "ben_12", "ben_13", "ben_14",   
            "ben_15", "ben_15_msoft", "ben_16", "ben_16_msoft", 
            "ben_17", "ben_17_msoft", 
            "ben_18_msoft", 
            "mal_19", "mal_19_msoft", "mal_20"]  

outFileName = out + feature_type + "_" + str(dim) + "D.pdf"
frames = []
perm_frames = []
refl_frames = []
native_frames = []

for ds in datasets:
    inputFile = "../datax/dynamic/csv/class_use/dcuf_" + ds + ".csv.gz"
    print(ds)
    df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
    if 'apkName' in df.columns: 
        df = df.drop('apkName', axis=1)
    frames.append(df)

for ds in datasets:
    inputFile = "../datax/dynamic/csv/perm_use/dpuf_" + ds + ".csv.gz"
    print(ds)
    df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
    if 'apkName' in df.columns: 
        df = df.drop('apkName', axis=1)
    perm_frames.append(df)

for ds in datasets:
    inputFile = "../datax/dynamic/csv/refl_use/druf_" + ds + ".csv.gz"
    print(ds)
    df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
    if 'apkName' in df.columns: 
        df = df.drop('apkName', axis=1)
    refl_frames.append(df)

for ds in datasets:
    inputFile = "../datax/dynamic/csv/native_use/dnuf_" + ds + ".csv.gz"
    print(ds)
    df = pd.read_csv(inputFile, compression='gzip', error_bad_lines=False, index_col=0)
    if 'apkName' in df.columns: 
        df = df.drop('apkName', axis=1)
    native_frames.append(df)

ddf = pd.concat(frames[:], ignore_index=True)
pdf = pd.concat(perm_frames[:], ignore_index=True)
rdf = pd.concat(refl_frames[:], ignore_index=True)
ndf = pd.concat(native_frames[:], ignore_index=True)

df = merge(ddf, pdf)
df = merge(df, ndf)
df = merge(df, rdf)

X = df.dropna(axis=1, how='all') # drop columns (axis=1) with 'all' NaN values
# get data without label
X = X.drop('classLabel', axis=1)
# y = labels
y = df['classLabel']

n_features = len(X.columns) # number of features
n_instances = len(X) # number of instances
print("No of features: " + str(n_features))
#y.head(10)

# convert to numpy arrays
X = X.values
y = y.values

# visualize original dataset
pca = PCA(n_components=n_components)
components = pca.fit_transform(df)

#pca = sklearnPCA() # consider all components
components = pca.fit_transform(X)
total_var = pca.explained_variance_ratio_.sum() * 100
labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'Ben=0, Mal=1'

fig = px.scatter_matrix(
    components,
    color=df['classLabel'],
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()
fig.write_image(outFileName)