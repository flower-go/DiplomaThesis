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
from transformers import WarmUp
from sklearn.metrics import f1_score

from text_classification_dataset import TextClassificationDataset

from sentiment_dataset import SentimentDataset


class Network:
    def __init__(self, args, labels):
        # vstup
        subwords = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        inp = [subwords]
        self.labels = labels

        # bert model
        config = transformers.AutoConfig.from_pretrained(args.bert)
        config.output_hidden_states = True
        self.bert = transformers.TFAutoModelForSequenceClassification.from_pretrained(args.bert, config=config)
        if args.freeze:
            self.bert.trainable = False

        # vezmu posledni vrstvu
        # TODO mohla bych vzít jen cls tokeny
        bert_output = self.bert(subwords, attention_mask=tf.cast(subwords != 0, tf.float32))[0]
        if args.freeze:
            print(str(args.freeze))
            bert_output.trainable = False
        dropout = tf.keras.layers.Dropout(args.dropout)(bert_output)
        predictions = tf.keras.layers.Dense(labels, activation=tf.nn.softmax)(dropout)

        self.model = tf.keras.Model(inputs=inp, outputs=predictions)
        self.optimizer=tf.optimizers.Adam()
        if args.warmup_decay > 0:
            self.optimizer.learning_rate = WarmUp(initial_learning_rate=args.epochs[0][1],warmup_steps=args.warmup_decay,
                                                   decay_schedule_fn=lambda step: 1 / math.sqrt(step))
        if args.model != None:
            self.model.load_weights(args.model)
        if args.label_smoothing:
            self.loss = tf.losses.CategoricalCrossentropy()
        else:
            self.loss = tf.losses.SparseCategoricalCrossentropy()
        self.metrics = {"loss": tf.metrics.Mean(), "F1": tf.metrics.Mean()}

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train_batch(self, inputs, gold_data, tvs):
        with tf.GradientTape() as tape:

            probabilities = self.model(inputs, training=True)
            tvs = tvs
            loss = 0.0


            if args.label_smoothing:
                loss += self.loss(tf.one_hot(gold_data, self.labels) * (1 - args.label_smoothing)
                    + args.label_smoothing /  self.labels, probabilities)
            else:
                loss += self.loss(tf.convert_to_tensor(gold_data), probabilities)

        gradients = tape.gradient(loss, tvs)

        tf.summary.experimental.set_step(self.optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self.metrics.items():
                metric.reset_states()
            self.metrics["loss"](loss)

            for name, metric in self.metrics.items():
                tf.summary.scalar("train/{}".format(name), metric.result())
        return gradients


    def train_epoch(self, dataset, args):
        num_gradients = 0
        tvs = self.model.trainable_variables
        print("trainable")
        print(str(len(tvs)))
        print(str(len(self.model.trainable_weights)))

        # if args.freeze:
        #     tvs = [tvar for tvar in tvs if not tvar.name.startswith('bert')]
        for batch in dataset.batches(size=args.batch_size):
            tg = self.train_batch(
                batch[0],
                batch[1], tvs)

            if not args.accu:
                self.optimizer.apply_gradients(zip(tg, tvs))
            else:
                if num_gradients == 0:
                    gradients = []
                    for index, g in enumerate(tg):
                        if g == None:
                            gradients.append(None)
                        elif not isinstance(g, tf.IndexedSlices):
                            gradients.append(g.numpy())
                        else:
                            gradients.append([(g.values.numpy(), g.indices.numpy())])

                else:
                    for g, ng in zip(gradients, tg):
                        if ng != None:
                            if isinstance(g, list):
                                g.append((ng.values.numpy(), ng.indices.numpy()))
                            else:
                                g += ng.numpy()
                num_gradients += 1
                if num_gradients == args.accu:
                    gradients = [tf.IndexedSlices(*map(np.concatenate, zip(*g))) if isinstance(g, list) else g for g in
                                 gradients]
                    if args.fine_lr > 0:
                        variables = self.model.trainable_variables
                        var1 = variables[0: args.lr_split]
                        var2 = variables[args.lr_split:]
                        tg1 = gradients[0: args.lr_split]
                        tg2 = gradients[args.lr_split:]

                        self._optimizer.apply_gradients(zip(tg2, var2))
                        self._fine_optimizer.apply_gradients(zip(tg1, var1))
                    else:
                        self.optimizer.apply_gradients(zip(gradients, tvs))
                    num_gradients = 0

    def train(self, data, args):
        for e, lr in args.epochs:
            if args.warmup_decay == 0:
                if args.accu > 0:
                    lr = lr / args.accu
            b.set_value(self.optimizer.learning_rate, lr)
            for i in range(e):
                print("epoch " + str(i))
                network.train_epoch(data.train, args)
                network.evaluate(data.dev, "dev", args)
                metrics = {name: metric.result() for name, metric in self.metrics.items()}
                metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
                print("Dev, epoch {}, lr {}, {}".format(i, lr, metrics_log))


    def predict(self, dataset, args):
        return self.model.predict(self._transform_dataset(dataset.data["tokens"]), batch_size=16)

    # def evaluate(self, dataset, name, args):
    #     return self.model.evaluate(self._transform_dataset(dataset.data["tokens"]), np.asarray(dataset.data["labels"]), 16)


    @tf.function(experimental_relax_shapes=True)
    def evaluate_batch(self, inputs, factors):
        probabilities = self.model(inputs, training=False)
        loss = 0


        if args.label_smoothing:
            loss += self.loss(tf.one_hot(factors, self.labels), probabilities)
        else:
            loss += self.loss(tf.convert_to_tensor(factors), probabilities)

        self.metrics["loss"](loss)
        pred = [np.argmax(p, axis=2) for p in probabilities]
        self.metrics["F1"](f1_score(factors,pred))

        return probabilities

    def evaluate(self, dataset, dataset_name, args):
        for metric in self.metrics.values():
            metric.reset_states()
        for batch in dataset.batches(size=args.batch_size):
            probabilities = self.evaluate_batch(batch[0], batch[1])

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
    parser.add_argument("--accu", default=0, type=int, help="accumulate batch size")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--datasets", default="facebook,csfd", type=str, help="Dataset for use")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
    parser.add_argument("--english", default=0, type=float, help="add some english data for training.")
    parser.add_argument("--epochs", default="10:5e-5,1:2e-5", type=str, help="Number of epochs.")
    parser.add_argument("--fine_lr", default=0, type=float, help="Learning rate for bert layers")
    parser.add_argument("--freeze", default=0, type=bool, help="Freezing bert layers")
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

    dataset = SentimentDataset(tokenizer)
    data_result = None
    data_other = None
    if args.datasets != None:
        for d in args.datasets.split(","):
            data = dataset.get_dataset(d,path="../../../datasets",debug=args.debug)
            if str(type(data)) != "<class 'text_classification_dataset.TextClassificationDataset'>":

                data_other = pd.concat([data_other, data])
            else:

                data_result = data


        if data_other is not None:
            train, test = train_test_split(data_other, test_size=0.3, shuffle=True, stratify=data_other["Sentiment"])
            dev, test = train_test_split(test, test_size=0.5, stratify=test["Sentiment"])
        if data_result == None:
            data_result = TextClassificationDataset().from_array([train,dev,test], tokenizer.encode)
        elif data_other is not None:
            data_other = TextClassificationDataset().from_array([train,dev,test], tokenizer.encode)
            data_result.append_dataset(data_other)

    if args.english > 0:
        imdb_ex, imdb_lab = dataset.get_dataset("imdb")
        imdb_ex = np.array(imdb_ex)
        imdb_lab = np.array(imdb_lab)
        imdb_ex, _,imdb_lab,_, = train_test_split(imdb_ex,imdb_lab, train_size=args.english, shuffle=True, stratify=imdb_lab)

        data_result.train._data["tokens"].append(imdb_ex)
        data_result.train._data["labels"].append(imdb_lab + 1)

    if args.warmup_decay > 0:
        args.warmup_decay = math.floor(len(data_result.train._data["tokens"]) / args.batch_size)

    print("Delka datasetu " + str(len(data_result.train._data)))

    # Create the network and train
    network = Network(args, len(data_result.train.LABELS))

    network.train(data_result, args)
    print("train ended")

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
    #data_result.train.save_mappings("{}/mappings.pickle".format(args.logdir))  # TODO
    if args.checkp:
        checkp = args.checkp
    else:
        checkp = args.logdir.split("/")[1]

    network.model.save_weights('./checkpoints/' + checkp)
    print(args.logdir.split("/")[1])

    if data_result.test.data["labels"][0] != -1:
        acc = (np.array(data_result.test.data["labels"]) == np.array(test_prediction))
        acc = sum(acc)/len(acc)
        print("Test accuracy: " + str(acc))



