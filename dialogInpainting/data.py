# Importing the necessary packages
import pandas as pd
import numpy as np
import os
import json
from random import randrange

class DataLoaderFunc:
  def __init__(self, path):
      self.path = path

  def csv_transform(a):
    #Function to transform a unified list to 3 separate lists
    l1,l2,l3 = [],[],[]
    for i in range(len(a)):
        l1.append(a[i][0][0])
        l2.append(a[i][1][0])
        l3.append(a[i][2][0])
    return l1,l2,l3
      
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
  
  def unzipper(a):
    # Function to unzip the list
    l1,l2 = [],[]
    for i in range(len(a)):
        l1.append(a[i][0][0])
        l2.append(a[i][1][0])
    return l1,l2
  
  def chang_series(a):
    # Function to add space to every conversation
    b = ""
    for i in a.values:
        b = b + i + " "
    return b
  
  def datagen(data, ids):
    #This function generates the data according to three word embedding approach
    counter=0
    new_lt = []
    for c in ids:
        conv = data["Conversation"][c]
        conv.replace(np.NaN, '', inplace=True)
        lst = []
        lt = []
        lst2 =[]
        s= ""
        sbar = ""
        r = randrange(1,len(conv)-1,2)
        if r!=(len(conv)-1):
            s = DataLoaderFunc.chang_series(conv[0:r]) + " " + "<extra_id>" + " " + DataLoaderFunc.chang_series(conv[r+1:len(conv)])
            sbar = DataLoaderFunc.chang_series(conv[0:len(conv)])
            lst.append(s)
            lst2.append(sbar)
        else:
            s = DataLoaderFunc.chang_series(conv[0:len(conv)-1]) + " " + "<extra_id>"
            sbar = DataLoaderFunc.chang_series(conv[0:len(conv)])
            lst.append(s)
            lst2.append(sbar)
        lt = [lst,lst2]
        cols = ["Train","Label"]
        new_lt.append(lt)
        counter = counter + 1
        print("Completed {0} : {1}".format(ids[counter],counter))

    a,b = DataLoaderFunc.unzipper(new_lt)
    out_data = pd.DataFrame(zip(a, b), columns =['Inputs', 'Labels'])
    out_data.head()
    out_data.to_csv('dialog_inpainting.csv',index = False)
  
##Calling the Classes
path = ("/content/Taskmaster/TM-3-2020/data")
load_data = DataLoaderFunc(path)
load_data.preprocess_data()
path_csv = "./out.csv"
data, ids = load_data.read_data(path_csv)
load_data.datagen(data,ids)