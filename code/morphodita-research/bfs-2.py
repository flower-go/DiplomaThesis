#!/usr/bin/env python3
import collections
import json
import math

import transformers
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import morpho_dataset
import pickle
import warnings
import morpho_dataset_simple as mds

from keras.models import load_model


# from transformers import WarmUp


class BertModel:
    def __init__(self, name, args):
        self.name = name
        self.config = transformers.BertConfig.from_pretrained(name)
        self.config.output_hidden_states = True
        self.tokenizer = transformers.BertTokenizer.from_pretrained(name)
        self.model = transformers.TFBertModel.from_pretrained(name,
                                                              config=self.config)
        self.embeddings_only = True if args.bert else False


class Network:
    def __init__(self, args, model, labels):

        self.factors = args.factors
        self._optimizer = tfa.optimizers.LazyAdam(beta_2=args.beta_2)
        # if args.warmup_decay > 0:
        #     self._optimizer.learning_rate = WarmUp(initial_learning_rate=args.epochs[0][1],warmup_steps=args.warmup_decay,
        #                                            decay_schedule_fn=lambda step: 1 / math.sqrt(step))

        segments = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        subwords = tf.keras.layers.Input(shape=[None], dtype=tf.int32)

        self.bert = model.model
        model_output = self.bert(subwords, attention_mask=tf.cast(subwords != 0, tf.float32))[2][-1]
        model_output = tf.keras.layers.Dense(labels, activation=tf.nn.softmax)(model_output)

        self.outer_model = tf.keras.Model(inputs=[subwords], outputs=model_output)


        if args.label_smoothing:
            self._loss = tf.losses.CategoricalCrossentropy()
        else:
            self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {"loss": tf.metrics.Mean()}
        for f in self.factors:
            self._metrics[f + "Raw"] = tf.metrics.SparseCategoricalAccuracy()
            self._metrics[f + "Dict"] = tf.metrics.Mean()

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    @tf.function(experimental_relax_shapes=True)
    def train_batch(self, inputs, factors):
        tags_mask = tf.not_equal(factors[0],0)
        with tf.GradientTape() as tape:
            probabilities = self.outer_model(inputs, training=True)
            tvs = self.outer_model.trainable_variables
            print(probabilities)
            print(str(probabilities.shape))

            loss = 0.0

            loss += self._loss(tf.convert_to_tensor(factors[0]), probabilities, tags_mask)

        gradients = tape.gradient(loss, tvs)

        tf.summary.experimental.set_step(self._optimizer.iterations)  # TODO  co to je?
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
            self._metrics["loss"](loss)


            self._metrics[self.factors[0] + "Raw"](factors[0], probabilities, tags_mask)
            for name, metric in self._metrics.items():
                tf.summary.scalar("train/{}".format(name), metric.result())
        return gradients

    def train_epoch(self, dataset, args, learning_rate):
        if args.warmup_decay == 0:
            self._optimizer.learning_rate = learning_rate

        num_gradients = 0

        while not dataset.train.epoch_finished():
            sentence_lens, batch = dataset.next_batch(args.batch_size)
            factors = []
            for f in self.factors:
                words = batch[dataset.FACTORS_MAP[f]].word_ids
                factors.append(words)
            inp = batch[dataset.FORMS].word_ids
            print('train epoch')

            tg = self.train_batch(inp, factors)

            if not args.accu:

                self._optimizer.apply_gradients(zip(tg, self.outer_model.trainable_variables))
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
                if num_gradients == args.accu or len(dataset.train._permutation) == 0:
                    gradients = [tf.IndexedSlices(*map(np.concatenate, zip(*g))) if isinstance(g, list) else g for g in
                                 gradients]
                    if args.fine_lr > 0:
                        variables = self.outer_model.trainable_variables
                        var1 = variables[0: args.lr_split]
                        var2 = variables[args.lr_split:]
                        tg1 = gradients[0: args.lr_split]
                        tg2 = gradients[args.lr_split:]

                        self._optimizer.apply_gradients(zip(tg2, var2))
                        self._fine_optimizer.apply_gradients(zip(tg1, var1))
                    else:
                        self._optimizer.apply_gradients(zip(gradients, self.outer_model.trainable_variables))
                    num_gradients = 0

    # TODO create inputs jako jednu metodu pro train i evaluate!
    # TODO vytvareni modelu jako jedna metoda pro outer i inner model
    def _compute_bert(self, batch, dataset, lenghts):

        # max_len = np.max([len(batch[dataset.BERT].word_ids[i]) for i in range(len(batch[dataset.BERT].word_ids))])
        # FIXME DATASET.BERT
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
        tags_mask = tf.not_equal(factors[0], 0)
        probabilities = self.outer_model(inputs, training=False)
        loss = 0

        loss += self._loss(tf.convert_to_tensor(factors[0]), probabilities[0], tags_mask)

        self._metrics["loss"](loss)
        self._metrics[self.factors[0] + "Raw"](factors[0], probabilities, tags_mask)

        return probabilities, tags_mask

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()
        while not dataset.epoch_finished():
            sentence_lens, batch = dataset.next_batch(args.batch_size)

            factors = []
            for f in self.factors:
                factors.append(batch[dataset.FACTORS_MAP[f]].word_ids)
            inp = [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs]

            probabilities, mask = self.evaluate_batch(inp, factors)

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
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
    # Fix random seed
    np.random.seed(42)
    tf.random.set_seed(42)

    command_line = " ".join(sys.argv[1:])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Input data")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    #parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    #parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    #parser.add_argument("--embeddings", default=None, type=str, help="External embeddings to use.")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    #parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    #parser.add_argument("--factors", default="Lemmas,Tags", type=str, help="Factors to predict.")
    #parser.add_argument("--factor_layers", default=1, type=int, help="Per-factor layers.")
    parser.add_argument("--label_smoothing", default=0.00, type=float, help="Label smoothing.")
    #parser.add_argument("--lemma_re_strip", default=r"(?<=.)(?:`|_|-[^0-9]).*$", type=str,
     #                   help="RE suffix to strip from lemma.")
    #parser.add_argument("--lemma_rule_min", default=2, type=int, help="Minimum occurences to keep a lemma rule.")
    # parser.add_argument("--min_epoch_batches", default=300, type=int, help="Minimum number of batches per epoch.")
    # parser.add_argument("--predict", default=None, type=str, help="Predict using the passed model.")
    # parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    # parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    # parser.add_argument("--rnn_layers", default=3, type=int, help="RNN layers.")
    # parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    # parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    # parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    parser.add_argument("--debug_mode", default=0, type=int, help="debug on small dataset")
    parser.add_argument("--bert", default=None, type=str, help="Bert model for embeddings")
    # parser.add_argument("--bert_model", default=None, type=str, help="Bert model for training")
    # parser.add_argument("--cont", default=0, type=int, help="load finetuned model and continue training?")
    parser.add_argument("--accu", default=0, type=int, help="accumulate batch size")
    # parser.add_argument("--test_only", default=None, type=str, help="Only test evaluation")
    # parser.add_argument("--fine_lr", default=0, type=float, help="Learning rate for bert layers")
    # parser.add_argument("--checkp", default=None, type=str, help="Checkpoint name")
    # parser.add_argument("--warmup_decay", default=0, type=int,
    #                     help="Number of warmup steps, than will be applied inverse square root decay")

    args = parser.parse_args()
    args.debug_mode = args.debug_mode == 1
#    args.cont = args.cont == 1

    # TODO vyřešit
    # tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    # tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    # tf.config.set_soft_device_placement(True)


    # Create logdir name
    if args.exp is None:
        args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    do_not_log = {"exp", "lemma_re_strip", "predict", "threads", "bert_model", "bert"}
    #TODO ma byt toto fakt zakomentovane?
    args.logdir = "models/{}".format(
        args.exp
        # ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key),
        #                          re.sub("[^,]*/", "", value) if type(value) == str else value)
        #           for key, value in sorted(vars(args).items()) if key not in do_not_log))
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



    args.bert_size = 768
    #if args.warmup_decay > 0:
    #    args.warmup_decay = math.floor(len(train.factors[1].word_strings) / args.batch_size)
    # Construct the network
    dataset = mds.SimpleDataset(args.debug_mode,args.data, args.bert)


    network = Network(args=args,
                      model=model_bert, labels=dataset.NUM_TAGS)

    log_file = open("{}/log".format(args.logdir), "w")
    #for factor in args.factors:
     #   print("{}: {}".format(factor, len(train.factors[train.FACTORS_MAP[factor]].words)), file=log_file,
       #       flush=True)
    print("Tagging with args:", "\n".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items())
                                           if key not in ["embeddings_data", "embeddings_words"])), flush=True)


    # def test_eval():
    #     metrics = network.evaluate(test, "test", args)
    #     metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
    #     for f in [sys.stderr, log_file]:
    #         print("Test, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f, flush=True)


    for i, (epochs, learning_rate) in enumerate(args.epochs):
        for epoch in range(epochs):

            network.train_epoch(dataset, args, learning_rate)

            # if dev:
            #     print("evaluate")
            #     metrics = network.evaluate(dev, "dev", args)
            #     metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
            #     for f in [sys.stderr, log_file]:
            #         print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f,
            #               flush=True)
            #
            # if args.cont and test:
            #     test_eval()

        #train.save_mappings("{}/mappings.pickle".format(args.logdir))  # TODO
        if args.checkp:
            checkp = args.checkp
        else:
            checkp = args.logdir.split("/")[1]

        network.outer_model.save_weights('./checkpoints/' + checkp)
        print(args.logdir.split("/")[1])

    # if test:
    #     test_eval()
