#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import transformers
from keras import backend as b
from sklearn.model_selection import train_test_split

from text_classification_dataset import TextClassificationDataset
import tensorflow_datasets as tfds


class Network:
    def __init__(self, args, labels):
        # TODO: Define a suitable model.

        # vstup
        subwords = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        inp = [subwords]

        config = transformers.BertConfig.from_pretrained(args.bert)
        config.output_hidden_states = True
        self.bert = transformers.TFBertForSequenceClassification.from_pretrained(args.bert,
                                                             config=config)

        bert_output = self.bert(subwords, attention_mask=tf.cast(subwords != 0, tf.float32))[0]
        # vezmu posledni vrstvu
        # dropout
        dropout = tf.keras.layers.Dropout(args.dropout)(bert_output)
        # dense s softmaxem
        predictions = tf.keras.layers.Dense(labels, activation=tf.nn.softmax)(bert_output)

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
        # TODO: Train the network on a given dataset.
        for e, lr in args.epochs:
            b.set_value(self.model.optimizer.learning_rate, lr)
            for i in range(e):
                network.train_epoch(omr.train, args)
                metrics = network.evaluate(omr.dev, "dev", args)


    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of the
        # predicted label indices (no probabilities/distributions).
        return self.model.predict(self._transform_dataset(dataset.data["tokens"]), batch_size=16)

    def evaluate(self, dataset, name, args):
        #zarovnat na stejnou delku a nacpat jako tensor nebo array
        return self.model.evaluate(self._transform_dataset(dataset.data["tokens"]), np.asarray(dataset.data["labels"]), 16)

    def _transform_dataset(self, dataset):
        max_len = max(len(a) for a in dataset)
        print(str(max_len))
        data = []
        for i in dataset:
            max_l = max_len - len(i)

            data.append(i + [0]*max_l)
        
        return np.asarray(data)



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--epochs", default="2:5e-5,1:2e-5", type=str, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
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

    # TODO: Create the BERT tokenizer to `tokenizer` variable
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)

    # TODO: Load the data, using a correct `tokenizer` argument, which
    # should be a callable that given a sentence in a string produces
    # a list/np.ndarray of token integers.

    def imdb_covertion(data):
        for i in range(len(data)):

            if len(data[i]) > 512:
                data[i] = data[i][0:512]

            data[i] = tokenizer.encode(data[i].decode('latin1'))
        return data

    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)
    train_data, train_labels = tfds.load(name="imdb_reviews", split="train",
                                      batch_size=-1, as_supervised=True)

    train_examples = tfds.as_numpy(train_data)
    train_examples = imdb_covertion(train_examples)

    facebook.train._data["tokens"].append(train_examples)
    facebook.train._data["labels"].append(np.array(train_labels) + 1)


    # Create the network and train
    network = Network(args, len(facebook.train.LABELS))
    network.train(facebook, args)

    # TODO pridat vyhodnoceni na dev!!!

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "sentiment_analysis_test.txt"
    test_prediction = []
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="ascii") as out_file:
        for label in network.predict(facebook.test, args):
            label = np.argmax(label)
            test_prediction.append(label)
            print(facebook.test.LABELS[label], file=out_file)

    if facebook.test.data["labels"][27] != -1:
        acc = (np.array(facebook.test.data["labels"]) == np.array(test_prediction))
        acc = sum(acc)/len(acc)
        print("Test accuracy: " + str(acc))


