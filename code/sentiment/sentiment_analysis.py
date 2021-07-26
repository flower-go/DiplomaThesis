

#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
import numpy as np
import tensorflow as tf
import transformers
import math
from keras import backend as b
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import WarmUp
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from text_classification_dataset import TextClassificationDataset

from sentiment_dataset import SentimentDataset


class Network:
    def __init__(self, args, labels):
        # vstup
        subwords = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        inp = [subwords]
        self.labels = labels

        # bert model
        if "robeczech" not in args.bert:
            config = transformers.AutoConfig.from_pretrained(args.bert)
            config.output_hidden_states = True
            self.bert = transformers.TFAutoModelForSequenceClassification.from_pretrained(args.bert, config=config)
        else:
            self.bert = transformers.TFAutoModelForSequenceClassification.from_pretrained(args.bert + "tf", output_hidden_states=True)
        if args.freeze:
            self.bert.trainable = False

        mask = tf.pad(subwords[:, 1:] != 0, [[0, 0], [1, 0]], constant_values=True)
        if args.layers == "att" and not args.freeze:
            bert_output = self.bert(subwords, attention_mask=tf.cast(mask, tf.float32))[1]
            weights = tf.Variable(tf.zeros([12]), trainable=True)
            output = 0
            softmax_weights = tf.nn.softmax(weights)
            for i in range(12):
                result = softmax_weights[i]*bert_output[i+1]
                output += result
        else:
            output = self.bert(subwords, attention_mask=tf.cast(mask, tf.float32))[1][-4:]
            output = tf.math.reduce_mean(
                output
                , axis=0)  # prumerovani vrstev
        output = tf.keras.layers.Dense(768, activation=tf.nn.tanh)(output[:, 0, :])
        dropout = tf.keras.layers.Dropout(args.dropout)(output)
        predictions = tf.keras.layers.Dense(labels, activation=tf.nn.softmax)(dropout)

        self.model = tf.keras.Model(inputs=inp, outputs=predictions)
        self.optimizer=tf.optimizers.Adam()
        if args.decay_type is not None:
            decay_steps = args.steps_in_epoch * (args.epochs[0][0] - args.warmup_decay)
            if args.decay_type == "i":
                initial_learning_rate = args.epochs[0][1]
                learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, decay_steps,
                                                                                 end_learning_rate=5e-5, power=0.5)
            elif args.decay_type == "c":
                learning_rate_fn = tf.keras.experimental.CosineDecay(args.epochs[0][1], decay_steps)
            self.optimizer.learning_rate = WarmUp(initial_learning_rate=args.epochs[0][1],
                                                   warmup_steps=args.warmup_decay * args.steps_in_epoch,
                                                   decay_schedule_fn=learning_rate_fn)
        if args.model != None:
            self.model.load_weights(args.model)
        if args.label_smoothing:
            self.loss = tf.losses.CategoricalCrossentropy()
        else:
            self.loss = tf.losses.SparseCategoricalCrossentropy()
        self.metrics = {"loss": tf.metrics.Mean(), "F1": tf.metrics.Mean()}

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    @tf.function(experimental_relax_shapes=True)
    def train_batch(self, inputs, gold_data, tvs):
        with tf.GradientTape() as tape:

            probabilities = self.model(inputs, training=True)
            tvs = tvs
            loss = 0.0

            #print("info")
            #print(str(self.labels))
            #print(str(len(probabilities)))
            #print(str(len(gold_data)))
            #print(str(len(inputs)))
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
        #print("trainable")
        #print(str(len(tvs)))
        #print(str(len(self.model.trainable_weights)))

        # if args.freeze:
        #     tvs = [tvar for tvar in tvs if not tvar.name.startswith('bert')]
        for batch in dataset.batches(size=args.batch_size):
            tf.print(batch[0])
            tg = self.train_batch(
                batch[0],
                batch[1], tvs)

            if args.accu < 2:
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
                tf.print("num gradients " + str(num_gradients))
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

                        self.optimizer.apply_gradients(zip(tg2, var2))
                        self.fine_optimizer.apply_gradients(zip(tg1, var1))
                    else:
                        self.optimizer.apply_gradients(zip(gradients, tvs))
                    num_gradients = 0

    def train(self, data, args):
        for e, lr in args.epochs:
            if args.decay_type is None:
                if args.accu > 1:
                    lr = lr / args.accu
                b.set_value(self.optimizer.learning_rate, lr)
            for i in range(e):
                print("epoch " + str(i))
                network.train_epoch(data.train, args)
                if args.kfold <= 0:
                    network.evaluate(data.dev, "dev", args)
                    metrics = {name: metric.result() for name, metric in self.metrics.items()}
                    metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
                    print("Dev, epoch {}, lr {}, {}".format(i, lr, metrics_log))


    def predict(self, dataset, args):
        if args.predict is None:
            data = dataset.data["tokens"]
        else:
            data = dataset
        return self.model.predict(self._transform_dataset(data), batch_size=16)

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


        return probabilities

    def evaluate(self, dataset, dataset_name, args):
        for metric in self.metrics.values():
            metric.reset_states()
        for batch in dataset.batches(size=args.batch_size):
            probabilities = self.evaluate_batch(batch[0], batch[1])
            pred = [np.argmax(p) for p in probabilities]
            self.metrics["F1"](f1_score(batch[1], pred, average="weighted"))

    def _transform_dataset(self, dataset):
        print(len(dataset))
        max_len = max(len(a) for a in dataset)
        print(max_len)
        print("max")
        data = []
        for i in dataset:
            print(i)
            print("i")

            max_l = max_len - len(i)
            data.append(i + [0]*max_l)
        
        res = np.asarray(data)
        #print("prevedeno")
        
        return res



def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--accu", default=1, type=int, help="accumulate batch size")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout.")
    parser.add_argument("--epochs", default="10:5e-5,1:2e-5", type=str, help="Number of epochs.")
    parser.add_argument("--layers", default=None, type=str, help="Which layers should be used")
    parser.add_argument("--warmup_decay", default=None, type=str,
                        help="Number of warmup steps, than will be applied inverse square root decay")
    parser.add_argument("--checkp", default=None, type=str, help="Checkpoint name")
    parser.add_argument("--debug", default=True, type=int, help="use small debug data")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--model", default=None, type=str, help="Model for loading")
    parser.add_argument("--predict", default=None, type=str, help="predict only from given file.")
    parser.add_argument("--datasets", default="csfd", type=str, help="Dataset for use")
    parser.add_argument("--english", default=0, type=float, help="add some english data for training.")
    parser.add_argument("--fine_lr", default=0, type=float, help="Learning rate for bert layers")
    parser.add_argument("--freeze", default=0, type=int, help="Freezing bert layers")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--kfold", default=None, type=str,
                        help="Number of folds for cross-validation and the index of the fold")
    args = parser.parse_args(args)
    args.epochs = [(int(epochs), float(lr)) for epochslr in args.epochs.split(",") for epochs, lr in
                   [epochslr.split(":")]]

    args.debug = args.debug == 1
    args.freeze = args.freeze == 1
    if args.kfold is not None:
        args.kfold = args.kfold.split(":")
        args.fold = args.kfold[1]
        args.kfold = args.kfold[0]
    else:
        args.kfold = 0

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    #    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    #    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if args.warmup_decay is not None:
        args.warmup_decay = args.warmup_decay.split(":")
        args.decay_type = args.warmup_decay[0]
        args.warmup_decay = int(args.warmup_decay[1])
    else:
        args.decay_type = None

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    if args.bert is not None and "robeczech" in args.bert:
        print("append path")
        sys.path.append(args.bert)
        import tokenizer.robeczech_tokenizer
    if not "robeczech" in args.bert:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)
    else:
        tokenizer = tokenizer.robeczech_tokenizer.RobeCzechTokenizer(args.bert + "tokenizer")

    if args.predict is None:

        dataset = SentimentDataset(tokenizer)
        data_result = None
        data_other = None
        if args.datasets != None:
            for d in args.datasets.split(","):
                data = dataset.get_dataset(d, path="../../../datasets", debug=args.debug)
                if str(type(data)) != "<class 'text_classification_dataset.TextClassificationDataset'>":
                    data["dataset"] = d
                    data_other = pd.concat([data_other, data])  # nedostane se sem None?
                else:

                    data_result = data

            if args.kfold > 0:
                len_data = 0
                all_data_tokens = []
                all_data_labels = []
                if data_other is not None:
                    len_o = len(data_other)
                    len_data += len_o
                    all_data_tokens.append(data_other["Post"])
                    all_data_labels.append(data_other["Sentiment"])
                if data_result is not None:
                    l_train = len(data_result.train._data["tokens"])
                    l_dev = len(data_result.dev._data["tokens"])
                    l_test = len(data_result.test._data["tokens"])
                    len_r = l_train + l_dev + l_test
                    len_data += len_r
                    all_data_tokens.append(data_result.train._data["tokens"])
                    all_data_tokens.append(data_result.dev._data["tokens"])
                    all_data_tokens.append(data_result.test._data["tokens"])
                    all_data_labels.append(data_result.train._data["labels"])
                    all_data_labels.append(data_result.dev._data["labels"])
                    all_data_labels.append(data_result.test._data["labels"])
                kf = KFold(n_splits=args.kfold)
                train, test = kf.split(np.array(range(len_data)))  # TODO resit ty jednotlive iterace
                train = train[args.fold]
                test = test[args.fold]

            if data_other is not None:
                if args.kfold > 0:
                    i_o = [i for i in train if i < len_o]
                    train = data_other[i_o]
                    i_o = [i for i in test if i < len_o]
                    test = data_other[i_o]
                    dev = []  # TODO doresit
                else:
                    train, test = train_test_split(data_other, test_size=0.3, shuffle=True,
                                                   stratify=data_other["Sentiment"])
                    dev, test = train_test_split(test, test_size=0.5, stratify=test["Sentiment"])
            # TODO docasny kod
            # with open("multitest", "w") as out_file:
            #    for index,l in test.iterrows():
            #        line = l["dataset"] + "\t" +  str(l["Sentiment"]) +  "\t" + l["Post"]
            #        print(line, file=out_file)
            if data_result == None:
                data_result = TextClassificationDataset().from_array([train, dev, test], tokenizer.encode)
            elif data_other is not None:
                data_other = TextClassificationDataset().from_array([train, dev, test], tokenizer.encode)
                if args.kfold <= 0:

                    data_result.append_dataset(data_other)
                else:
                    i_train = [i for i in train if i < l_train]
                    i_dev = [i for i in train if i < l_dev]
                    i_test = [i for i in train if i < l_test]

                    # train
                    data_result.train._data["tokens"] = data_result.train._data["tokens"][i_train]
                    data_result.train._data["labels"] = data_result.train._data["labels"][i_train]
                    data_result.train._data["tokens"].append(data_result.dev._data["tokens"][i_dev])
                    data_result.train._data["labels"].append(data_result.dev._data["labels"][i_dev])
                    data_result.train._data["tokens"].append(data_result.train._data["tokens"][i_test])
                    data_result.train._data["labels"].append(data_result.train._data["labels"][i_test])

                    # test
                    i_train = [i for i in test if i < l_train]
                    i_dev = [i for i in test if i < l_dev]
                    i_test = [i for i in test if i < l_test]

                    data_result.test._data["tokens"] = data_result.train._data["tokens"][i_train]
                    data_result.test._data["labels"] = data_result.train._data["labels"][i_train]
                    data_result.test._data["tokens"].append(data_result.dev._data["tokens"][i_dev])
                    data_result.test._data["labels"].append(data_result.dev._data["labels"][i_dev])
                    data_result.test._data["tokens"].append(data_result.train._data["tokens"][i_test])
                    data_result.test._data["labels"].append(data_result.train._data["labels"][i_test])

        if args.english > 0:
            imdb_ex, imdb_lab = dataset.get_dataset("imdb")
            imdb_ex = np.array(imdb_ex)
            imdb_lab = np.array(imdb_lab)
            print("delka imdb")
            print(len(imdb_lab))
            print(len(imdb_ex))
            if args.english < 1:
                print("less than one")
                size = min(len(data_result.train._data["tokens"]) * args.english, len(imdb_ex)) / len(imdb_ex)
                if size < 1:
                    imdb_ex, _, imdb_lab, _, = train_test_split(imdb_ex, imdb_lab, train_size=size, shuffle=True,
                                                                stratify=imdb_lab)

                data_result.train._data["tokens"].append(imdb_ex)
                data_result.train._data["labels"].append(imdb_lab + 1)
            else:  # zero shot
                size = len(imdb_ex)
                data_result.train._data["tokens"] = imdb_ex
                data_result.train._data["labels"] = imdb_lab + 1
                test_labels_new = []
                test_data_new = []
                for i in range(len(data_result.test._data["labels"])):
                    if data_result.test._data["labels"] != 0:
                        test_data_new.append(data_result.test._data["tokens"][i])
                        test_labels_new.append(data_result.test._data["labels"][i])

                data_result.test._data["tokens"] = test_data_new
                data_result.test._data["labels"] = test_labels_new

                # print(type(data_result.test._data["labels"]))

                data_result.test._size = len(test_data_new)
                data_result.train._size = len(imdb_ex)
        num_labels = len(data_result.train.LABELS)

    else:
        num_labels = 3

        # if args.decay_type is not None:
        #   args.warmup_decay = math.floor(len(data_result.train._data["tokens"]) / args.batch_size)

        # print("Delka datasetu " + str(len(data_result.train._data)))

    if args.decay_type != None:
        args.steps_in_epoch = math.floor(len(data_result.train._data["tokens"]) / (args.batch_size * args.accu))
    # Create the network and train
    network = Network(args, num_labels)

    if args.predict is None:
        network.train(data_result, args)

        # Generate test set annotations, but to allow parallel execution, create it
        # in in args.logdir if it exists.
        # TODO vypisovani i s textem a gold label! ale jen když chci
        out_path = "sentiment_analysis_test.txt"
        test_prediction = []
        if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
        with open(out_path, "w", encoding="ascii") as out_file:
            for label in network.predict(data_result.test, args):
                label = np.argmax(label)
                test_prediction.append(label)
                print(data_result.test.LABELS[label], file=out_file)
        # data_result.train.save_mappings("{}/mappings.pickle".format(args.logdir))  # TODO
        if args.checkp:
            checkp = args.checkp
        else:
            checkp = args.logdir.split("/")[1]

        network.model.save_weights('./checkpoints/' + checkp)
        print(args.logdir.split("/")[1])

        if data_result.test.data["labels"][0] != -1:
            acc = (np.array(data_result.test.data["labels"]) == np.array(test_prediction))
            acc = sum(acc) / len(acc)
            c = confusion_matrix(np.array(data_result.test.data["labels"]), np.array(test_prediction))
            print(c)
            print("Test accuracy: " + str(acc))

            print("F1 metrics: " + str(
                f1_score(np.array(data_result.test.data["labels"]), np.array(test_prediction), average="weighted")))

    else:
        # TODO do args.model dat co nacist a do predict asi teda data k predikci
        out_file = args.predict + "_vystup"
        # TODO nacist test file

        data = pd.read_csv(args.predict, sep='\n', header=None, names=['Post']).assign(Sentiment=4)
        test = []
        for i, row in data.iterrows():
            text = row["Post"].rstrip("\r\n")[0:512]
            encoded = tokenizer.encode(text)
            if type(encoded) is dict:
                encoded = encoded["input_ids"]
            test.append(encoded)

        with open(out_file, "w") as out_file:
            for i, label in enumerate(network.predict(test, args)):
                label = np.argmax(label)
                print("post")
                print(data.iloc[i]["Post"])
                line = str(label) + "\t" + data.iloc[i]["Post"]
                print(line, file=out_file)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
    # Parse arguments

