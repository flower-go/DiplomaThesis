
#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np

import tensorflow as tf
import transformers
import math
from keras import backend as b
import pandas as pd
from sklearn.model_selection import train_test_split
#from transformers import WarmUp
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from text_classification_dataset import TextClassificationDataset

from sentiment_dataset import SentimentDataset


if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--accu", default=0, type=int, help="accumulate batch size")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--datasets", default="mall,facebook,csfd", type=str, help="Dataset for use")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
    parser.add_argument("--english", default=0, type=float, help="add some english data for training.")
    parser.add_argument("--epochs", default="10:5e-5,1:2e-5", type=str, help="Number of epochs.")
    parser.add_argument("--fine_lr", default=0, type=float, help="Learning rate for bert layers")
    parser.add_argument("--freeze", default=0, type=int, help="Freezing bert layers")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--model", default=None, type=str, help="Model for loading")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--debug", default=False, type=int, help="use small debug data")
    parser.add_argument("--checkp", default=None, type=str, help="Checkpoint name")
    parser.add_argument("--warmup_decay", default=0, type=int, help="Number of warmup steps, than will be applied inverse square root decay")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.epochs = [(int(epochs), float(lr)) for epochslr in args.epochs.split(",") for epochs, lr in
                   [epochslr.split(":")]]

    args.debug = args.debug == 1
    args.freeze = args.freeze == 1

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    #    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    #    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    dataset = SentimentDataset(None)
    data_result = None
    if args.datasets != None:
        for d in args.datasets.split(","):
            data = dataset.get_dataset(d, path="../../../datasets", debug=args.debug)
            print("Dataset " + d)
            print(data.head())
            data_result = pd.concat([data_result, data])


    # if data_result is not None:
    #     res_x = np.concatenate((np.array(data_result.train._data["tokens"]),np.array(data_result.dev._data["tokens"]),np.array(data_result.test._data["tokens"])))
    #     res_y = np.concatenate((np.array(data_result.train._data["labels"]),np.array(data_result.dev._data["labels"]),np.array(data_result.test._data["labels"])))
    #     data_result = pd.DataFrame({"Post": res_x, "Sentiment": res_y})
    #
    # if args.english > 0:
    #     imdb_ex, imdb_lab = dataset.get_dataset("imdb")
    #     imdb_ex = np.array(imdb_ex)
    #     imdb_lab = np.array(imdb_lab)
    #     imdb_ex, _, imdb_lab, _, = train_test_split(imdb_ex, imdb_lab, train_size=args.english, shuffle=True,
    #                                                 stratify=imdb_lab)
    #
    #     data_result.train._data["tokens"].append(imdb_ex)
    #     data_result.train._data["labels"].append(imdb_lab + 1)
    #
    # ##preprocessings bag of word model
    # from sklearn.feature_extraction.text import CountVectorizer
    #
    # vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    # #print(data_result["Post"])
    # X = vectorizer.fit_transform(data_result["Post"]).toarray()
    #
    #
    # from sklearn.feature_extraction.text import TfidfTransformer
    #
    # tfidfconverter = TfidfTransformer()
    # X = tfidfconverter.fit_transform(X).toarray()
    # X_train, X_test, y_train, y_test = train_test_split(X, data_result["Sentiment"], test_size=0.2, random_state=0, stratify=data_result["Sentiment"])
    #
    #
    # #Training data
    # classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    # classifier.fit(X_train, y_train)
    #
    # y_pred = classifier.predict(X_test)
    #
    # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    #
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))




