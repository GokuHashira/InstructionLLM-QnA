# Importing the necessary packages
import pandas as pd
import numpy as np

class DataLoaderFunc:
  def __init__(self, path):
      self.path = path

  def preprocess_data(self):
    # This function handles the preprocessing of the data
    data = pd.read_csv(self.path)
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
path = ("/content/gdrive/MyDrive/Data/out.csv")
load_data = DataLoaderFunc(path)
data, ids = load_data.preprocess_data()
load_data.datagen(data,ids)