#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import transformers
from keras import backend as b
import pandas as pd
from sklearn.model_selection import train_test_split

from text_classification_dataset import TextClassificationDataset

from sentiment_dataset import SentimentDataset


class Network:
    def __init__(self, args, labels):
        # vstup
        subwords = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        inp = [subwords]

        # bert model
        config = transformers.AutoConfig.from_pretrained(args.bert)
        config.output_hidden_states = True
        self.bert = transformers.TFAutoModelForSequenceClassification.from_pretrained(args.bert, config=config)

        # vezmu posledni vrstvu
        bert_output = self.bert(subwords, attention_mask=tf.cast(subwords != 0, tf.float32))[0]
        dropout = tf.keras.layers.Dropout(args.dropout)(bert_output)
        # dense s softmaxem
        predictions = tf.keras.layers.Dense(labels, activation=tf.nn.softmax)(dropout)

        self.model = tf.keras.Model(inputs=inp, outputs=predictions)
        # compile model
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(size=args.batch_size):
            metrics = self.model.train_on_batch(
                batch[0],
                batch[1],
                reset_metrics=True)

            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            with self._writer.as_default():
                for name, value in zip(self.model.metrics_names, metrics):
                    tf.summary.scalar("train/{}".format(name), value)

    def train(self, omr, args):
        for e, lr in args.epochs:
            b.set_value(self.model.optimizer.learning_rate, lr)
            for i in range(e):
                network.train_epoch(omr.train, args)
                metrics = network.evaluate(omr.dev, "dev", args)
                print("Dev, epoch {}, lr {}, {}".format(i, lr, metrics[1]))


    def predict(self, dataset, args):
        return self.model.predict(self._transform_dataset(dataset.data["tokens"]), batch_size=16)

    def evaluate(self, dataset, name, args):
        return self.model.evaluate(self._transform_dataset(dataset.data["tokens"]), np.asarray(dataset.data["labels"]), 16)

    def _transform_dataset(self, dataset):
        max_len = max(len(a) for a in dataset)
        data = []
        for i in dataset:
            max_l = max_len - len(i)
            data.append(i + [0]*max_l)
        
        res = np.asarray(data)
        print("prevedeno")
        
        return res



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--epochs", default="10:5e-5,1:2e-5", type=str, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--english", default=0, type=float, help="add some english data for training.")
    parser.add_argument("--datasets", default="facebook,csfd", type=str, help="Dataset for use")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.epochs = [(int(epochs), float(lr)) for epochslr in args.epochs.split(",") for epochs, lr in
                   [epochslr.split(":")]]

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    #    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    #    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)

    # Load data



    dataset = SentimentDataset(tokenizer)
    data_result = None
    data_other = None
    if args.datasets != None:
        for d in args.datasets.split(","):
            data = dataset.get_dataset(d,path="../../../datasets")
            if str(type(data)) != "<class 'text_classification_dataset.TextClassificationDataset'>":

                data_other = pd.concat([data_other, data])
            else:
                data_result = data


        if data_other is not None:
            train, test = train_test_split(data_other, test_size=0.3, shuffle=True, stratify=data_other["Sentiment"])
            dev, test = train_test_split(test, test_size=0.5, stratify=test["Sentiment"])
        if data_result == None:
            data_result = TextClassificationDataset().from_array(data_other, tokenizer.encode)
        elif data_other is not None:

            data_other = TextClassificationDataset().from_array([train,dev,test], tokenizer.encode)
            print("pridani train " + str(len(data_other.train._data["tokens"])))
            #TODO tokenize
            data_result.append_dataset(data_other)
            #data_result.train.append_data(data_other.train._data["tokens"],data_other.train._data["labels"])

            # data_result.train._data["tokens"].append(data_other.train._data["tokens"])
            # data_result.train._data["labels"].append(np.array(data_other.train._data["labels"]))
            #
            # data_result.dev._data["tokens"].append(data_other.dev._data["tokens"])
            # data_result.dev._data["labels"].append(np.array(data_other.dev._data["labels"]))
            #
            # data_result.test._data["tokens"].append(data_other.test._data["tokens"])
            # data_result.test._data["labels"].append(np.array(data_other.test._data["labels"]))



    if args.english > 0:
        imdb_ex, imdb_lab = dataset.get_dataset("imdb")
        imdb_ex = np.array(imdb_ex)
        imdb_lab = np.array(imdb_lab)
        imdb_ex, _,imdb_lab,_, = train_test_split(imdb_ex,imdb_lab, train_size=args.english, shuffle=True, stratify=imdb_lab)

        data_result.train._data["tokens"].append(imdb_ex)
        data_result.train._data["labels"].append(imdb_lab + 1)


    # Create the network and train
    network = Network(args, len(data_result.train.LABELS))
    network.train(data_result, args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "sentiment_analysis_test.txt"
    test_prediction = []
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="ascii") as out_file:
        for label in network.predict(data_result.test, args):
            label = np.argmax(label)
            test_prediction.append(label)
            print(data_result.test.LABELS[label], file=out_file)

    if data_result.test.data["labels"][27] != -1:
        acc = (np.array(data_result.test.data["labels"]) == np.array(test_prediction))
        acc = sum(acc)/len(acc)
        print("Test accuracy: " + str(acc))


