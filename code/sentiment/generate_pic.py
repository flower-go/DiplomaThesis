#!/usr/bin/env python
# coding: utf-8

# In[35]:


import argparse
import datetime
import os
import re

import numpy as np
#import tensorflow as tf
import transformers
from keras import backend as b
from sklearn.model_selection import train_test_split

from text_classification_dataset import TextClassificationDataset
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
labels = {'n': 1, '0': 0, 'p': 2, 'b': 'BIP'}
my_col =  ["#DC267F","#FFB000","#648FFF"]
# obecna funkce
def metriky(name,data):
    print(pd.unique(data["Sentiment"]))
    print(str(len(data)))
    neutral = sum(np.array(data["Sentiment"]) == labels['0'])
    positive = sum(np.array(data["Sentiment"]) == labels['p'])
    negative = sum(np.array(data["Sentiment"]) == labels['n'])
    names = [ 'negative', 'neutral', 'positive']
    values = [negative, neutral, positive]

    plt.figure()


    plt.bar(names, values, color=my_col)
    plt.title(name + " dataset responses distribution")
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(values):
        plt.text(xlocs[i], v, str(v))
    plt.show()
    return names, values


# In[54]:



path = "../../../datasets"
def nacti(directory):
    return pd                 .concat([
                pd.read_csv('{}/positive.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=labels['p']),
                pd.read_csv('{}/neutral.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=labels['0']),
                pd.read_csv('{}/negative.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=labels['n'])
            ], axis=0) \
                .reset_index(drop=True)
    


# ## Mall

# In[91]:




directory = path + "/mallcz"
mall = nacti(directory)

mall_names, mall_values = metriky("mall", mall)


# ## CSFD

# In[96]:


directory = path + "/csfd"
csfd = nacti(directory)

csfd_names, csfd_values = metriky("mall", csfd)


# ## facebook

# In[159]:


from text_classification_dataset import TextClassificationDataset
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
facebook = TextClassificationDataset(path + "/czech_facebook", tokenizer=tokenizer.encode)

print("delka cekem" + str(len(facebook.train._data["tokens"]) + len(facebook.dev._data["tokens"]) + len(facebook.test._data["tokens"])))
data = []
for d in ["train","test","dev"]:
    data.extend(getattr(facebook,d)._data["labels"])

facebook_values = [sum(np.array(data)==1), sum(np.array(data)==0), sum(np.array(data)==2)]
    

    


# In[158]:


sum(np.array(data)==1)


# In[163]:


fig, axs = plt.subplots(2, 3,sharex=True, figsize=(15,5))
fig.suptitle('Distribution of classes across datasets')

values = [mall_values,csfd_values,facebook_values]
axs[0,0].bar(mall_names, mall_values,color=my_col)
axs[0,0].set_title("mall")

axs[0,1].bar(csfd_names,csfd_values,color=my_col)
axs[0,1].set_title("csfd")
axs[0,2].bar(csfd_names,facebook_values,color=my_col)
axs[0,2].set_title("facebook")



    
gs = axs[1, 2].get_gridspec()
# remove the underlying axes
for ax in axs[1, 0:]:
    ax.remove()
axbig = fig.add_subplot(gs[1, 0:])
#axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
 #              xycoords='axes fraction', va='center')

# Plot bars and create text labels for the table
cell_text = []
for row in range(3):
    cell_text.append(values[row]) 
cell_text = np.array(cell_text).transpose()
axbig.axis('tight')
axbig.set_axis_off() 
the_table = axbig.table(cellText=cell_text,
                      rowLabels=mall_names,
                      colLabels=["mall","csfd","facebook"],
                       loc="center")
plt.savefig("distribution.pdf")

def metriky_all(name,data):
    print(str(len(data)))
    neutral = data[1]
    positive = data[2]
    negative = data[0]
    names = [ 'negative', 'neutral', 'positive']
    values = [negative, neutral, positive]

    fig1, ax1 = plt.subplots(figsize=(15,5))
    ax1.axis('equal') 
    


    my_col = ["#DC267F","#FFB000","#648FFF"]
    _, _, autotexts = plt.pie(values,labels=names,colors=my_col, explode=[0.01, 0.01, 0.01],autopct='%1.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
    plt.title(name + " dataset responses distribution")
    plt.tight_layout()
    plt.show()

name = "all"
data = []
datasety = [mall_values, facebook_values, csfd_values]
for i in range(3):
    celkem = 0
    for j in range(3):
        celkem = celkem + datasety[j][i]
    data.append(celkem)
    
metriky_all(name, data)

