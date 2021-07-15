import os
import pandas as pd
from utils import Jaccard_Similarity,cosine
import pickle

import csv        


f_output =  open('Task B_predictions_EN.tsv', 'w', newline='')
tsv_output = csv.writer(f_output, delimiter='\t')
tsv_output.writerow(["Sl. No.","Instance Id","Predicted Label"])


# load the model from disk
model = pickle.load(open('knnpickle_file', 'rb'))


path = "Files/"
test = path + "test/"

x_test = []
for id,folder in enumerate(os.listdir(test)):
    fol_path = test + folder + "/"
    docs = []
    for file in os.listdir(fol_path):
            docs.append(open(fol_path + file,"r",encoding="utf-8").read())
    val = [Jaccard_Similarity(docs[0],docs[1]),cosine(docs[0],docs[1])]
    res = model.predict([val])
    tsv_output.writerow([id+1,folder,res[0]])

f_output.close()