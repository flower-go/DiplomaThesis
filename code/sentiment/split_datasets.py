import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
dataset = "mallcz"
path = "../../../datasets/"

np.random.seed(42)



def split_data(directory):
    data_all= pd \
                    .concat([
                    pd.read_csv('{}/positive.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                        Sentiment=2),
                    pd.read_csv('{}/neutral.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                        Sentiment=0),
                    pd.read_csv('{}/negative.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                        Sentiment=1)
                ], axis=0) \
                    .reset_index(drop=True)

    train, test = train_test_split(data_all, test_size=0.3, shuffle=True, stratify=data_all["Sentiment"])
    dev, test = train_test_split(test, test_size=0.5, stratify=test["Sentiment"])

    test_posts = test["Post"]

    with open(dataset + "_test", "w") as out_file:
        for l in test_posts:
            print(l, file=out_file)


for dataset in ["mallcz, csfd"]:
    directory = path + dataset
    split_data(directory)






