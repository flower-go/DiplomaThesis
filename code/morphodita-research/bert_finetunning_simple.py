#!/usr/bin/env python3
import sys

import collections
import json
import math

import transformers
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as b
import morpho_dataset
import morpho_dataset_simple as mds
import pickle
import warnings
from transformers import WarmUp

from keras.models import load_model


# from transformers import WarmUp


class BertModel:
    def __init__(self, name, args):
        self.name = name

        if "rob" in name:
            self.path = name
            self.tokenizer = tokenizer.robeczech_tokenizer.RobeCzechTokenizer(self.path + "tokenizer")
            self.model = transformers.TFAutoModel.from_pretrained(self.path + "tf", output_hidden_states=True)
        else:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(name)
            self.config = transformers.BertConfig.from_pretrained(name)
            self.config.output_hidden_states = True
            self.model = transformers.TFAutoModel.from_pretrained(name,
                                                              config=self.config)
        self.embeddings_only = True if args.bert else False                                                     


class Network:
    def __init__(self, args, model, labels, num_chars):



        subwords = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        charseq_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)


        cle = tf.keras.layers.Embedding(num_chars, args.cle_dim, mask_zero=True)(charseqs)
        cle = tf.keras.layers.Dropout(rate=args.dropout)(cle)
        cle = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim), merge_mode="concat")(cle)
        cle = tf.gather(1 * cle, charseq_ids)


        inp = [subwords, charseqs, charseq_ids]
        bert = model.model
        #TODO dopsat att
        mask = tf.pad(subwords[:, 1:] != 0, [[0, 0], [1, 0]], constant_values=True)
        if args.layers == "att":
            bert_output = bert(subwords, attention_mask=tf.cast(mask, tf.float32))[2]
            weights = tf.Variable(tf.zeros([12]), trainable=True)
            output = 0
            softmax_weights = tf.nn.softmax(weights)
            for i in range(12):
                result = softmax_weights[i] * bert_output[i + 1]
                output += result
            bert_output = output
        else:
            bert_output = bert(subwords, attention_mask=tf.cast(mask, tf.float32))[0]
        self.labels = labels
        dropout = tf.keras.layers.Dropout(rate=args.dropout)(bert_output)

        inputs = [dropout,cle]
        hidden = tf.keras.layers.Concatenate()(inputs)
        hidden = tf.keras.layers.Dropout(rate=args.dropout)(hidden)
        #TODO co s tim teď?
        # dense s softmaxem
        predictions_tags = tf.keras.layers.Dense(labels[1], activation=tf.nn.softmax)(dropout)
        predictions_lemmas = tf.keras.layers.Dense(labels[0], activation=tf.nn.softmax)(hidden)
        out = [predictions_lemmas, predictions_tags]
        # model(inputs, outputs)
        self.model = tf.keras.Model(inputs=inp, outputs=out)

        if args.model != None:
            self.model.load_weights(args.model)
        # compile model
        self.optimizer=tf.optimizers.Adam()
        if args.decay_type is not None:
            if args.decay_type == "i":
                initial_learning_rate = args.epochs[0][1]
                decay_steps = 1.0
                decay_rate = 0.5
                learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps,
                                                                                  decay_rate)
            elif args.decay_type == "c":
                decay_steps = args.epochs[0][0] * (args.steps_in_epoch * - args.warmup_decay)
                learning_rate_fn = tf.keras.experimental.CosineDecay(args.epochs[0][1], decay_steps)

            self.optimizer.learning_rate = WarmUp(initial_learning_rate=args.epochs[0][1],
                                                   warmup_steps=args.warmup_decay * args.steps_in_epoch,
                                                   decay_schedule_fn=learning_rate_fn)
        if args.label_smoothing:
            self.loss = tf.losses.CategoricalCrossentropy()
        else:
            self.loss = tf.losses.SparseCategoricalCrossentropy()
        self.metrics = {"loss": tf.metrics.Mean()}
        for f in args.factors:
            self.metrics[f + "Raw"] = tf.metrics.SparseCategoricalAccuracy()
        self.metrics["LemmasTagsRaw"] = tf.metrics.Mean()

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    @tf.function(experimental_relax_shapes=True)
    def train_batch(self, inputs, factors):
        tags_mask = tf.pad(inputs[0][:, 1:] != 0, [[0, 0], [1, 0]], constant_values=True)
        tf.print("factors")
        tf.print(inputs[0],summarize=-1)
        tf.print("masky")
        tf.print(tags_mask, summarize=-1)
        with tf.GradientTape() as tape:

            probabilities = self.model(inputs, training=True)
            tvs = self.model.trainable_variables
            if args.freeze:
                tvs = [tvar for tvar in tvs if not tvar.name.startswith('bert')]
            loss = 0.0

            for i in range(len(args.factors)):
                if args.label_smoothing:
                    loss += self.loss(tf.one_hot(factors[i], self.labels[i]) * (1 - args.label_smoothing)
                        + args.label_smoothing /  self.labels[i], probabilities[i], tags_mask)
                else:
                    loss += self.loss(tf.convert_to_tensor(factors[i]), probabilities[i], tags_mask)

        gradients = tape.gradient(loss, tvs)

        tf.summary.experimental.set_step(self.optimizer.iterations)  # TODO  co to je?
        with self._writer.as_default():
            for name, metric in self.metrics.items():
                metric.reset_states()
            self.metrics["loss"](loss)
            for i in range(len(args.factors)):
                self.metrics[args.factors[i] + "Raw"](factors[i], probabilities[i], tags_mask)

            for name, metric in self.metrics.items():
                tf.summary.scalar("train/{}".format(name), metric.result())
        return gradients

    def train_epoch(self, dataset, args, learning_rate):
        if args.decay_type is None:
            self.optimizer.learning_rate = learning_rate
            if args.accu>0:                                                             
                self.optimizer.learning_rate = self.optimizer.learning_rate/args.accu

        num_gradients = 0

        while not dataset.epoch_finished():
            sentence_lens, batch = dataset.next_batch(args.batch_size, args.word_dropout)
            factors = []
            for f in args.factors:
                words = batch[dataset.data.FACTORS_MAP[f]].word_ids
                factors.append(words)
            #print("tvar " + str(batch[dataset.data.FACTORS_MAP["Lemmas"]].word_ids.shape))
            #print("kolik je maskovanych " + str(np.sum(batch[dataset.data.FACTORS_MAP["Lemmas"]].word_ids == 0, axis=1)))
            #print("kolik neni " + str(np.sum(batch[dataset.data.FACTORS_MAP["Lemmas"]].word_ids != 0, axis=1)))
            inp = [batch[dataset.data.FORMS].word_ids, batch[dataset.data.FORMS].charseqs,batch[dataset.data.FORMS].charseq_ids,]
            #print("WI", batch[dataset.data.FORMS].word_ids)
            #print("CS", batch[dataset.data.FORMS].charseqs)
            #print("CSI", batch[dataset.data.FORMS].charseq_ids, flush=True)
            print('train epoch')

            #TODO neměla bych prumerovat metriky?

            tg = self.train_batch(inp, factors)

            if not args.accu:
                self.optimizer.apply_gradients(zip(tg, self.model.trainable_variables))
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

                    # gradients = [
                    #   g.numpy() if not isinstance(g, tf.IndexedSlices) else [(g.values.numpy(), g.indices.numpy())] for g
                    #   in tg]
                else:

                    for g, ng in zip(gradients, tg):
                        if ng != None:
                            if isinstance(g, list):
                                g.append((ng.values.numpy(), ng.indices.numpy()))
                            else:
                                g += ng.numpy()
                num_gradients += 1
                if num_gradients == args.accu or len(dataset.data._permutation) == 0:
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
                        print("trainable variables")
                        print(str(len(self.model.trainable_variables)))
                        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    num_gradients = 0

    # TODO create inputs jako jednu metodu pro train i evaluate!
    # TODO vytvareni modelu jako jedna metoda pro outer i inner model
    def _compute_bert(self, batch, dataset, lenghts):

        max_len = batch[dataset.EMBEDDINGS].word_ids.shape[1]
        result = np.zeros((len(batch[dataset.BERT].word_ids), max_len, len(batch[dataset.BERT].word_ids[0][0])))
        for sentence in range(len(batch[dataset.BERT].word_ids)):
            result[sentence][0:len(batch[dataset.BERT].word_ids[sentence])] = batch[dataset.BERT].word_ids[sentence]

        return result

    def _compute_embeddings(self, batch, dataset):
        embeddings = np.zeros([batch[dataset.EMBEDDINGS].word_ids.shape[0],
                               batch[dataset.EMBEDDINGS].word_ids.shape[1],
                               args.embeddings_size])
        for i in range(embeddings.shape[0]):
            for j in range(embeddings.shape[1]):
                if batch[dataset.EMBEDDINGS].word_ids[i, j]:
                    embeddings[i, j] = args.embeddings_data[batch[dataset.EMBEDDINGS].word_ids[i, j] - 1]
        return embeddings

    # TODO predelat, kdyz uz to vraci rovnou embeddings
    def _compute_bert_embeddings(self, batch, dataset):
        bert_embeddings = np.zeros([batch[dataset.BERT].word_ids.shape[0],
                                    batch[dataset.BERT].word_ids.shape[1],
                                    args.bert_size])
        for i in range(bert_embeddings.shape[0]):
            for j in range(bert_embeddings.shape[1]):
                if batch[dataset.BERT].word_ids[i, j]:
                    bert_embeddings[i, j] = dataset.bert_embeddings[batch[dataset.BERT].word_ids[i, j] - 1]

        return bert_embeddings

    @tf.function(experimental_relax_shapes=True)
    def evaluate_batch(self, inputs, factors):
        tags_mask =  tf.pad(factors[0][:, 1:] != 0, [[0, 0], [1, 0]], constant_values=True)
        #print("mask")
        #tf.print(tags_mask)
        probabilities = self.model(inputs, training=False)
        loss = 0

        for i in range(len(args.factors)):
            if args.label_smoothing:
                loss += self.loss(tf.one_hot(factors[i], self.labels[i]), probabilities[i],
                                  tags_mask)
            else:
                loss += self.loss(tf.convert_to_tensor(factors[i]), probabilities[i], tags_mask)

        self.metrics["loss"](loss)
        for i in range(len(args.factors)):
            self.metrics[args.factors[i] + "Raw"](factors[i], probabilities[i], tags_mask)

        return probabilities, tags_mask

    def evaluate(self, dataset, dataset_name, args):
        for metric in self.metrics.values():
            metric.reset_states()
        while not dataset.epoch_finished():
            sentence_lens, batch = dataset.next_batch(args.batch_size)

            factors = []
            for f in args.factors:
                words = batch[dataset.data.FACTORS_MAP[f]].word_ids
                factors.append(words)
            inp = [batch[dataset.data.FORMS].word_ids, batch[dataset.data.FORMS].charseqs, batch[dataset.data.FORMS].charseq_ids,]


            probabilities, mask = self.evaluate_batch(inp, factors)


            if len(args.factors) == 2:
                predictions_raw = [np.argmax(p, axis=2) for p in probabilities]
                self.metrics["LemmasTagsRaw"](
                np.logical_and(factors[0] == predictions_raw[0], factors[1] == predictions_raw[1]), mask)
        metrics = {name: metric.result() for name, metric in self.metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics


if __name__ == "__main__":
    import argparse
    import datetime
    import json
    import os
    import sys
    import re

    print(os.getcwd())

    #TODO pridat jako parametr
    np.random.seed(42)
    tf.random.set_seed(42)

    command_line = " ".join(sys.argv[1:])

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--accu", default=0, type=int, help="accumulate batch size")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--bert", default=None, type=str, help="Bert model for embeddings")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    parser.add_argument("--checkp", default=None, type=str, help="Checkpoint name")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cont", default=0, type=int, help="load finetuned model and continue training?")
    parser.add_argument("--debug_mode", default=0, type=int, help="debug on small dataset")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--factors", default="Lemmas,Tags", type=str, help="Factors to predict.")
    parser.add_argument("--fine_lr", default=0, type=float, help="Learning rate for bert layers")
    parser.add_argument("--freeze", default=0, type=bool, help="Freezing bert layers")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--model", default=None, type=str, help="Model for loading")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--warmup_decay", default=None, type=str,help="Number of warmup steps, than will be applied inverse square root decay")
    parser.add_argument("--word_dropout", default=0, type=float, help="Word dropout rate")
    parser.add_argument("data", type=str, help="Input data")
    parser.add_argument("--layers", default=None, type=str, help="Which layers should be used")

    args = parser.parse_args()
    args.debug_mode = args.debug_mode == 1
    args.freeze = args.freeze == 1
    args.cont = args.cont == 1

    if args.bert is not None and "rob" in args.bert:
        sys.path.append(args.bert)
        import tokenizer.robeczech_tokenizer

    if args.warmup_decay is not None:
        args.warmup_decay = args.warmup_decay.split(":")
        args.decay_type = args.warmup_decay[0]
        args.warmup_decay = int(args.warmup_decay[1])
    else:
        args.decay_type = None

    # TODO vyřešit
    # tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    # tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    # tf.config.set_soft_device_placement(True)


    # Create logdir name
    if args.exp is None:
        args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    do_not_log = {"exp", "lemma_re_strip", "predict", "threads", "bert_model", "bert"} #TODO vyřešit

    args.logdir = "models/{}".format(
        args.exp
    )
    if not os.path.exists("models"): os.mkdir("models")
    if not os.path.exists(args.logdir): os.mkdir(args.logdir)

    # Dump passed options
    with open("{}/options.json".format(args.logdir), mode="w") as options_file:
        json.dump(vars(args), options_file, sort_keys=True)


    # Postprocess args
    args.factors = args.factors.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in
                   (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]
    model_bert = BertModel(args.bert, args)

    dataset = mds.SimpleDataset(args.debug_mode, args.data,"train", model_bert)

    dev = mds.SimpleDataset(args.debug_mode,args.data, "dev", model_bert, train=dataset.data)

    test = mds.SimpleDataset(args.debug_mode, args.data, "test", model_bert, train=dataset.data)

    args.bert_size = 768
    #if args.warmup_decay > 0:
    #    args.warmup_decay = math.floor(len(train.factors[1].word_strings) / args.batch_size)
    # Construct the network

    #train_encodings = train.bert_subwords
    #train_tag_labels = train._factors[train.TAGS].word_ids
    #train_segments = train.bert_segments
    #labels_unique = len(train.factors[train.TAGS].words)
    if args.decay_type != None:
        args.steps_in_epoch = math.floor(len(dataset.data.factors[1].word_strings) / args.batch_size)
    network = Network(args=args,
                      model=model_bert, labels=[dataset.NUM_LEMMAS,dataset.NUM_TAGS], num_chars=dataset.num_chars)

    # TODO nemame predikci !
    # slova: batch[0].word_ids
    # tags a lemmas: batch[dataset.FACTORS_MAP[f]].word_ids
    # zakodovane:                 batch[dataset.SEGMENTS].word_ids
    #              batch[dataset.SUBWORDS].word_ids

    log_file = open("{}/log".format(args.logdir), "w")
    for factor in args.factors:
        print("{}: {}".format(factor, len(dataset.data.factors[dataset.data.FACTORS_MAP[factor]].words)), file=log_file,
              flush=True)
    print("Tagging with args:", "\n".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items())
                                           if key not in ["embeddings_data", "embeddings_words"])), flush=True)


    def test_eval():
        metrics = network.evaluate(test, "test", args)
        metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
        for f in [sys.stderr, log_file]:
            print("Test, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f, flush=True)


    for i, (epochs, learning_rate) in enumerate(args.epochs):
        for epoch in range(epochs):
            print("train epoch {}".format(epoch))
            network.train_epoch(dataset, args, learning_rate)

            if dev:
                print("evaluate")
                metrics = network.evaluate(dev, "dev", args)
                metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
                for f in [sys.stderr, log_file]:
                    print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f,
                          flush=True)

            if args.cont and test:
                test_eval()

        dataset.data.save_mappings("{}/mappings.pickle".format(args.logdir))  # TODO
        if args.checkp:
            checkp = args.checkp
        else:
            checkp = args.logdir.split("/")[1]

        network.model.save_weights('./checkpoints/' + checkp)
        print(args.logdir.split("/")[1])

    if test:
        test_eval()
