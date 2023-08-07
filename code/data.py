# Importing the necessary packages
import pandas as pd
import numpy as np
import csv
import json
import os

class DataLoaderFunc:
  def __init__(self, path):
      self.path = path

  def csv_transform(a):
      l1,l2,l3 = [],[],[]
      for i in range(len(a)):
        l1.append(a[i][0][0])
        l2.append(a[i][1][0])
        l2.append(a[i][2][0])
      return l1,l2

  def preprocess_data(self, path):
        # This function is responsible for creating csv file from the json files
        speaker = []
        conv = []
        convID = []
        lst = []
        json_file_names = [filename for filename in os.listdir(self.path) if filename.endswith('.json')]
        for json_file_name in json_file_names:
            with open(os.path.join(self.path, json_file_name)) as json_file:
                data = json.load(f)
                print("Number of conversations: ",len(data))
                for i in range(len(data)):
                    for j in range(len(data[i]['utterances'])):
                        speaker.append(data[i]['utterances'][j]['speaker'])
                        conv.append(data[i]['utterances'][j]['text'])
                        convID.append(data[i]["conversation_id"])
        lst = [speaker,conv,convID]
        cols = ["Speaker","Conversation","ConversationID"]
        s,c,ci = DataLoaderFunc.csv_transform(lst)
        out_data = pd.DataFrame(zip(lst), columns =cols)
        out_data.head()
        out_data.to_csv('out.csv', index=False)

  def read_data(path_to_csv):
    # This function handles the preprocessing of the data
    data = pd.read_csv(path_to_csv)
    print(data.shape)
    print("Number of conversations: ",len(data["ConversationID"].unique()))
    ids = data["ConversationID"].unique()
    new_data = data.set_index('ConversationID')
    return new_data, ids
  
  def datagen(data, ids):
    #This function generates the data according to three word embedding approach
    counter=0
    lst = []
    for c in ids:
        conv = data["Conversation"][c]
        conv.replace(np.NaN, '', inplace=True)
        s = ""
        for i in range(0,len(conv),2):
            s= ""
            if i <= len(conv)-3:
                s = s + conv[i] + " " + "<extra_id>" + " " + conv[i+2]
                lst.append(s)
            if len(conv)%3!=0:
                if i > len(conv)-3:
                    s = s + conv[i] + " " + "<extra_id>"
                    lst.append(s)
        counter = counter + 1
        print("Completed {0} : {1}".format(ids[counter],counter))

    counter=0
    lst2 = []
    for c in ids:
        conv = data["Conversation"][c]
        conv.replace(np.NaN, '', inplace=True)
        s = ""
        for i in range(0,len(conv),2):
            s= ""
            if i <= len(conv)-3:
                s = s + conv[i] + " " + conv[i+1] + " " + conv[i+2]
                lst2.append(s)
            if len(conv)%3!=0:
                if i > len(conv)-3:
                    s = s + conv[i] + " " + conv[i]
                    lst2.append(s)
        counter = counter + 1
        print("Completed {0} : {1}".format(ids[counter],counter))

    new_data = pd.DataFrame(list(zip(lst, lst2)),columns =['Inputs', 'Labels'])
    new_data.head()
    new_data.to_csv(r'/content/three_sentenced_data.csv')
  
##Calling the Classes
path = ("/content/Taskmaster/TM-3-2020/data")
load_data = DataLoaderFunc(path)
load_data.preprocess_data()
path_csv = "./out.csv"
data, ids = load_data.read_data(path_csv)
load_data.datagen(data,ids)