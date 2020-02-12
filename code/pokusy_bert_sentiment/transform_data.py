from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from tqdm import tqdm_notebook
import utils


import functools

import numpy as np
import tensorflow as tf

prefix = '../../../twitter-airline-sentiment/'


def convert():
    prefix = '../../../twitter-airline-sentiment/'


    train_df = pd.read_csv(prefix + 'Tweets.csv')
    print(train_df.head())
    train_df = train_df[['text','airline_sentiment']]
    train_df.head()
    train_df['airline_sentiment'] = train_df['airline_sentiment'].replace({'positive':1, 'neutral':0, 'negative':2})
    train_df = pd.DataFrame({
        'idx':range(len(train_df)),
        'label':train_df['airline_sentiment'],
        'alpha': ['a'] * train_df.shape[0],
        'txt': train_df['text'].replace(r'\n', ' ', regex=True)
    })

    train_df.to_csv(prefix + 'train.tsv', sep='\t', index=False, header=False, columns=train_df.columns)
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(train_df['idx'].values, tf.int32),
                tf.cast(train_df['sentence'].values, tf.string),
                tf.cast(train_df['label'].values, tf.int32)
            )
        )
    )
    return training_dataset

def just_get():


    train_df = pd.read_csv(prefix + 'Tweets.csv')
    print(train_df.head())
    train_df = train_df[['text', 'airline_sentiment']]
    train_df.head()
    train_df['airline_sentiment'] = train_df['airline_sentiment'].replace({'positive': 1, 'neutral': 0, 'negative': 2})
    train_df = pd.DataFrame({
        'guid': range(len(train_df)),
        'text_a': train_df['text'].replace(r'\n', ' ', regex=True),
        'text_b': ['a'] * train_df.shape[0],
        'label': train_df['airline_sentiment']
    })

    return train_df

def get_items():
    lines = utils.DataProcessor._read_tsv(prefix + 'train.tsv')
    proc = utils.BinaryProcessor()
    examples = proc._create_examples(lines,'train')
    return examples

get_items()
#print("ahoj")