import os
import pandas as pd
from utils import Jaccard_Similarity,cosine

path = "Files/"
train = path + "train/"
dev = path + "dev/"


train_labels = pd.read_csv(path + "EN_train_task_c.tsv", sep='\t', header=0)
name_template = train + "task_C_en_train_"


train_df = []
for id, row in train_labels.iterrows():
    if row["Task"] != "C":
        continue
    key = "{:04d}".format(int(row["Directory"]))
    fol_path = name_template + key +"/"
    docs = []
    val = []
    for file in os.listdir(fol_path):
            docs.append(open(fol_path + file,"r",encoding="utf-8").read())

    val.append(fol_path)
    val.append(Jaccard_Similarity(docs[0],docs[1]))
    val.append(cosine(docs[0],docs[1]))
    val.append(row["Label"])
    train_df.append(val)
    

dev_labels = pd.read_csv(path + "EN_dev_task_c.tsv", sep='\t', header=0)
name_template = dev + "task_C_en_dev_"


dev_df = []
for id, row in dev_labels.iterrows():
    if row["Task"] != "C":
        continue
    key = "{:04d}".format(int(row["Directory"]))
    fol_path = name_template + key +"/"
    docs = []
    val = []
    for file in os.listdir(fol_path):
            docs.append(open(fol_path + file,"r",encoding="utf-8").read())

    val.append(fol_path)
    val.append(Jaccard_Similarity(docs[0],docs[1]))
    val.append(cosine(docs[0],docs[1]))
    val.append(row["Label"])
    dev_df.append(val)


x_train = []
y_train = []

for val in train_df:
    x_train.append(val[1:3])
    y_train.append(val[3])
    

x_dev = []
y_dev = []

for val in dev_df:
    x_dev.append(val[1:3])
    y_dev.append(val[3])
    

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

neighbors = KNeighborsClassifier(n_neighbors=7)
neighbors.fit(x_train, y_train)

predictions = neighbors.predict(x_dev)
print("Accuracy: ",accuracy_score(y_dev, predictions))
print(classification_report(y_dev,predictions))

knnPickle = open('knnpickle_file_C', 'wb') 
pickle.dump(neighbors, knnPickle)                      
