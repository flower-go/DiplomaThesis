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
    def __init__(self, args, num_words, num_chars, factor_words, model, labels):

        self.factors = args.factors
        self.factor_words = factor_words
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
        with tf.GradientTape() as tape:
            probabilities = self.outer_model(inputs, training=True)
            tvs = self.outer_model.trainable_variables
            print(probabilities)
            print(str(probabilities.shape))

            if len(self.factors) == 1:
                probabilities = [probabilities]
            loss = 0.0
            for i in range(len(self.factors)):
                if args.label_smoothing:
                    loss += self._loss(
                        tf.one_hot(factors[i], self.factor_words[self.factors[i]]) * (1 - args.label_smoothing)
                        + args.label_smoothing / self.factor_words[self.factors[i]], probabilities[i],
                        probabilities[i]._keras_mask)
                else:
                    loss += self._loss(tf.convert_to_tensor(factors[i]), probabilities[i], probabilities[i]._keras_mask)

        gradients = tape.gradient(loss, tvs)

        tf.summary.experimental.set_step(self._optimizer.iterations)  # TODO  co to je?
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
            self._metrics["loss"](loss)
            for i in range(len(self.factors)):
                self._metrics[self.factors[i] + "Raw"](factors[i], probabilities[i], probabilities[i]._keras_mask)
            for name, metric in self._metrics.items():
                tf.summary.scalar("train/{}".format(name), metric.result())
        return gradients

    def train_epoch(self, dataset, args, learning_rate):
        if args.warmup_decay == 0:
            self._optimizer.learning_rate = learning_rate

        num_gradients = 0

        while not train.epoch_finished():
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
                if num_gradients == args.accu or len(train._permutation) == 0:
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
        probabilities = self.outer_model(inputs, training=False)
        if len(self.factors) == 1:
            probabilities = [probabilities]
        loss = 0
        for i in range(len(self.factors)):
            if args.label_smoothing:
                loss += self._loss(tf.one_hot(factors[i], self.factor_words[self.factors[i]]), probabilities[i],
                                   probabilities[i]._keras_mask)
            else:
                loss += self._loss(tf.convert_to_tensor(factors[i]), probabilities[i], probabilities[i]._keras_mask)

        self._metrics["loss"](loss)
        for i in range(len(self.factors)):
            self._metrics[self.factors[i] + "Raw"](factors[i], probabilities[i], probabilities[i]._keras_mask)

        return probabilities, [probabilities[f]._keras_mask for f in range(len(self.factors))]

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()
        while not dataset.epoch_finished():
            sentence_lens, batch = dataset.next_batch(args.batch_size)

            factors = []
            for f in self.factors:
                factors.append(batch[dataset.FACTORS_MAP[f]].word_ids)
            any_analyses = any(batch[train.FACTORS_MAP[factor]].analyses_ids for factor in self.factors)
            inp = [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs]
            if args.embeddings:
                embeddings = self._compute_embeddings(batch, dataset)
                inp.append(embeddings)

            if args.bert:
                bert_embeddings = self._compute_bert(batch, dataset, sentence_lens)
                inp.append(bert_embeddings)

            if args.bert_model:
                inp.append(batch[dataset.SEGMENTS].word_ids)
                inp.append(batch[dataset.SUBWORDS].word_ids)

            probabilities, mask = self.evaluate_batch(inp, factors)

            if any_analyses:
                predictions = [np.argmax(p, axis=2) for p in probabilities]

                for i in range(len(sentence_lens)):
                    for j in range(sentence_lens[i]):

                        analysis_ids = [batch[dataset.FACTORS_MAP[factor]].analyses_ids[i][j] for factor in
                                        self.factors]
                        if not analysis_ids or len(analysis_ids[0]) == 0:
                            continue

                        known_analysis = any(all(analysis_ids[f][a] != dataset.UNK for f in range(len(self.factors)))
                                             for a in range(len(analysis_ids[0])))
                        if not known_analysis:
                            continue

                        # Compute probabilities of unknown analyses as minimum probability
                        # of a known analysis - 1e-3.

                        analysis_probs = [probabilities[factor][i, j].numpy() for factor in range(len(self.factors))]

                        for f in range(len(args.factors)):
                            min_probability = None
                            for analysis in analysis_ids[f]:

                                # TODO spatny shape, probabilities f analysis je vektor
                                if analysis != dataset.UNK and (min_probability is None or analysis_probs[f][
                                    analysis] - 1e-3 < min_probability):
                                    min_probability = analysis_probs[f][analysis] - 1e-3

                            analysis_probs[f][dataset.UNK] = min_probability
                            analysis_probs[f][dataset.PAD] = min_probability

                        best_index, best_prob = None, None
                        for index in range(len(analysis_ids[0])):
                            prob = sum(analysis_probs[f][analysis_ids[f][index]] for f in range(len(args.factors)))
                            if best_index is None or prob > best_prob:
                                best_index, best_prob = index, prob
                        for f in range(len(args.factors)):
                            predictions[f][i, j] = analysis_ids[f][best_index]

            for fc in range(len(self.factors)):
                self._metrics[self.factors[fc] + "Dict"](factors[fc] == predictions[fc],
                                                         mask[fc])

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
    parser.add_argument("--char_dropout", default=0, type=float, help="Character dropout")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--embeddings", default=None, type=str, help="External embeddings to use.")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--factors", default="Lemmas,Tags", type=str, help="Factors to predict.")
    parser.add_argument("--factor_layers", default=1, type=int, help="Per-factor layers.")
    parser.add_argument("--label_smoothing", default=0.00, type=float, help="Label smoothing.")
    parser.add_argument("--lemma_re_strip", default=r"(?<=.)(?:`|_|-[^0-9]).*$", type=str,
                        help="RE suffix to strip from lemma.")
    parser.add_argument("--lemma_rule_min", default=2, type=int, help="Minimum occurences to keep a lemma rule.")
    parser.add_argument("--min_epoch_batches", default=300, type=int, help="Minimum number of batches per epoch.")
    parser.add_argument("--predict", default=None, type=str, help="Predict using the passed model.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=3, type=int, help="RNN layers.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    parser.add_argument("--debug_mode", default=0, type=int, help="debug on small dataset")
    parser.add_argument("--bert", default=None, type=str, help="Bert model for embeddings")
    parser.add_argument("--bert_model", default=None, type=str, help="Bert model for training")
    parser.add_argument("--cont", default=0, type=int, help="load finetuned model and continue training?")
    parser.add_argument("--accu", default=0, type=int, help="accumulate batch size")
    parser.add_argument("--test_only", default=None, type=str, help="Only test evaluation")
    parser.add_argument("--fine_lr", default=0, type=float, help="Learning rate for bert layers")
    parser.add_argument("--checkp", default=None, type=str, help="Checkpoint name")
    parser.add_argument("--warmup_decay", default=0, type=int,
                        help="Number of warmup steps, than will be applied inverse square root decay")

    args = parser.parse_args()
    args.debug_mode = args.debug_mode == 1
    args.cont = args.cont == 1

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


    # Load input data
    data_paths = [None] * 3
    if args.debug_mode:
        print("DEBUG MODE")
        data_paths[0] = "{}-train-small.txt".format(args.data)
        data_paths[1] = "{}-dev-small.txt".format(args.data)
        data_paths[2] = "{}-test-small.txt".format(args.data)
    else:
        data_paths[0] = "{}-train.txt".format(args.data)
        data_paths[1] = "{}-dev.txt".format(args.data)
        data_paths[2] = "{}-test.txt".format(args.data)

    # TODO udelat cyklus
    train = morpho_dataset.MorphoDataset(data_paths[0],
                                         embeddings=args.embeddings_words if args.embeddings else None,
                                         bert=model_bert,
                                         lemma_re_strip=args.lemma_re_strip,
                                         lemma_rule_min=args.lemma_rule_min, simple=True)

    if os.path.exists(data_paths[1]):
        dev = morpho_dataset.MorphoDataset(data_paths[1], train=train, shuffle_batches=False,
                                           bert=model_bert, simple=True
                                           )
    else:
        dev = None

    if os.path.exists(data_paths[2]):
        test = morpho_dataset.MorphoDataset(data_paths[2], train=train, shuffle_batches=False,
                                            bert=model_bert, simple=True
                                            )
    else:
        test = None

    args.bert_size = 768
    if args.warmup_decay > 0:
        args.warmup_decay = math.floor(len(train.factors[1].word_strings) / args.batch_size)
    # Construct the network

    train_encodings = train.bert_subwords
    train_tag_labels = train._factors[train.TAGS].word_ids
    train_segments = train.bert_segments
    labels_unique = len(train.factors[0].words_map) #TODO tohle je nekde ulozené


    def encode_tags(tags, encodings, segments):
        labels = tags
        encoded_labels = []
        for doc_labels, s in zip(labels, segments):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(s), dtype=int) * -100
            first_indices = np.nonzero(np.r_[1, np.diff(s)])

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[first_indices] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels


    train_labels = encode_tags(train_tag_labels, train_encodings,train_segments )


    network = Network(args=args,
                      num_words=len(train.factors[train.FORMS].words),
                      num_chars=len(train.factors[train.FORMS].alphabet),
                      factor_words=dict(
                          (factor, len(train.factors[train.FACTORS_MAP[factor]].words)) for factor in args.factors),
                      model=model_bert, labels=len(labels_unique))

    # TODO nemame predikci !
    # slova: batch[0].word_ids
    # tags a lemmas: batch[dataset.FACTORS_MAP[f]].word_ids
    # zakodovane:                 batch[dataset.SEGMENTS].word_ids
    #              batch[dataset.SUBWORDS].word_ids

    log_file = open("{}/log".format(args.logdir), "w")
    for factor in args.factors:
        print("{}: {}".format(factor, len(train.factors[train.FACTORS_MAP[factor]].words)), file=log_file,
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

            network.train_epoch(train, args, learning_rate)

            if dev:
                print("evaluate")
                metrics = network.evaluate(dev, "dev", args)
                metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
                for f in [sys.stderr, log_file]:
                    print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f,
                          flush=True)

            if args.cont and test:
                test_eval()

        train.save_mappings("{}/mappings.pickle".format(args.logdir))  # TODO
        if args.checkp:
            checkp = args.checkp
        else:
            checkp = args.logdir.split("/")[1]

        network.outer_model.save_weights('./checkpoints/' + checkp)
        print(args.logdir.split("/")[1])

    if test:
        test_eval()
