#!/usr/bin/env python3
import collections
import json

import numpy as np
import tensorflow as tf

import morpho_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads,
                                                                       allow_soft_placement=True))

    def construct(self, args, num_words, num_chars, factor_words, predict_only):
        self.METRICS = []
        for factors in [["Lemmas"], ["Tags"], ["Lemmas", "Tags"]]:
            for use_dict in ["Raw", "Dict"]:
                if all(factor in args.factors for factor in factors):
                    self.METRICS.append("".join(factors) + use_dict)

        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])
            if args.embeddings: self.embeddings = tf.placeholder(tf.float32, [None, None, args.embeddings_size])
            if args.elmo_size: self.elmo = tf.placeholder(tf.float32, [None, None, args.elmo_size])
            self.factors = dict((factor, tf.placeholder(tf.int32, [None, None])) for factor in args.factors)
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])

            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Word embeddings
            inputs = []
            if args.we_dim:
                word_embeddings = tf.get_variable("word_embeddings", shape=[num_words, args.we_dim], dtype=tf.float32)
                inputs.append(tf.nn.embedding_lookup(word_embeddings, self.word_ids))

            # Character-level embeddings
            character_embeddings = tf.get_variable("character_embeddings", shape=[num_chars, args.cle_dim], dtype=tf.float32)
            characters_embedded = tf.nn.embedding_lookup(character_embeddings, self.charseqs)
            characters_embedded = tf.layers.dropout(characters_embedded, rate=args.dropout, training=self.is_training)
            _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim), tf.nn.rnn_cell.GRUCell(args.cle_dim),
                characters_embedded, sequence_length=self.charseq_lens, dtype=tf.float32)
            cle = tf.concat([state_fwd, state_bwd], axis=1)
            cle_inputs = tf.nn.embedding_lookup(cle, self.charseq_ids)
            # If CLE dim is half WE dim, we add them together, which gives
            # better results; otherwise we concatenate CLE and WE.
            if 2 * args.cle_dim == args.we_dim:
                inputs[-1] += cle_inputs
            else:
                inputs.append(cle_inputs)

            # Pretrained embeddings
            if args.embeddings:
                inputs.append(self.embeddings)

            # Contextualized embeddings
            if args.elmo_size:
                inputs.append(self.elmo)

            # All inputs done
            inputs = tf.concat(inputs, axis=2)

            # RNN layers
            hidden_layer = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    hidden_layer, sequence_length=self.sentence_lens, dtype=tf.float32,
                    scope="word-level-rnn-{}".format(i))
                previous = hidden_layer
                hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                if i: hidden_layer += previous

            # Tagger
            loss = 0
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            weights_sum = tf.reduce_sum(weights)
            self.predictions, self.prediction_probs = {}, {}
            for factor in args.factors:
                factor_layer = hidden_layer
                for _ in range(args.factor_layers):
                    factor_layer += tf.layers.dropout(tf.layers.dense(factor_layer, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                if factor == "Lemmas": factor_layer = tf.concat([factor_layer, cle_inputs], axis=2)
                output_layer = tf.layers.dense(factor_layer, factor_words[factor])
                self.predictions[factor] = tf.argmax(output_layer, axis=2, output_type=tf.int32)
                self.prediction_probs[factor] = tf.nn.softmax(output_layer, axis=2)

                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.factors[factor], factor_words[factor]) * (1 - args.label_smoothing) + args.label_smoothing / factor_words[factor]
                    loss += tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                else:
                    loss += tf.losses.sparse_softmax_cross_entropy(self.factors[factor], output_layer, weights=weights)

            # Pretrain saver
            self.saver_inference = tf.train.Saver(max_to_keep=1)
            if predict_only: return

            # Training
            self.global_step = tf.train.create_global_step()
            self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=self.global_step)

            # Train saver
            self.saver_train = tf.train.Saver(max_to_keep=1)

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [
                    tf.contrib.summary.scalar("train/loss", loss),
                    tf.contrib.summary.scalar("train/lr", self.learning_rate)]
                for factor in args.factors:
                    self.summaries["train"].append(tf.contrib.summary.scalar(
                        "train/{}".format(factor),
                        tf.reduce_sum(tf.cast(tf.equal(self.factors[factor], self.predictions[factor]), tf.float32) * weights) /
                        weights_sum))

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=weights_sum)
                self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
                self.metrics = dict((metric, tf.placeholder(tf.float32, [])) for metric in self.METRICS)
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss)]
                    for metric in self.METRICS:
                        self.summaries[dataset].append(tf.contrib.summary.scalar("{}/{}".format(dataset, metric),
                                                                                 self.metrics[metric]))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, learning_rate, args):
        batches, at_least_one_epoch = 0, False
        while batches < args.min_epoch_batches:
            while not train.epoch_finished():
                sentence_lens, b = train.next_batch(args.batch_size)
                if args.word_dropout:
                    mask = np.random.binomial(n=1, p=args.word_dropout, size=b[train.FORMS].word_ids.shape)
                    b[train.FORMS].word_ids = (1 - mask) * b[train.FORMS].word_ids + mask * train.UNK
                if args.char_dropout:
                    mask = np.random.binomial(n=1, p=args.char_dropout, size=b[train.FORMS].charseqs.shape)
                    b[train.FORMS].charseqs = (1 - mask) * b[train.FORMS].charseqs + mask * train.UNK

                feeds = {self.is_training: True, self.learning_rate: learning_rate, self.sentence_lens: sentence_lens,
                         self.charseqs: b[train.FORMS].charseqs, self.charseq_lens: b[train.FORMS].charseq_lens,
                         self.word_ids: b[train.FORMS].word_ids, self.charseq_ids: b[train.FORMS].charseq_ids}
                if args.embeddings:
                    if args.word_dropout:
                        mask = np.random.binomial(n=1, p=args.word_dropout, size=b[train.EMBEDDINGS].word_ids.shape)
                        b[train.EMBEDDINGS].word_ids = (1 - mask) * b[train.EMBEDDINGS].word_ids
                    embeddings = np.zeros([b[train.EMBEDDINGS].word_ids.shape[0], b[train.EMBEDDINGS].word_ids.shape[1], args.embeddings_size])
                    for i in range(embeddings.shape[0]):
                        for j in range(embeddings.shape[1]):
                            if b[train.EMBEDDINGS].word_ids[i, j]:
                                embeddings[i, j] = args.embeddings_data[b[train.EMBEDDINGS].word_ids[i, j] - 1]
                    feeds[self.embeddings] = embeddings
                if args.elmo_size:
                    feeds[self.elmo] = b[train.ELMOS].word_ids
                for factor in args.factors:
                    feeds[self.factors[factor]] = b[train.FACTORS_MAP[factor]].word_ids
                self.session.run([self.training, self.summaries["train"]], feeds)
                batches += 1
                if at_least_one_epoch: break
            at_least_one_epoch = True

    def predict(self, dataset, output, args):
        evaluating = not output
        if evaluating:
            metrics, words = collections.OrderedDict((metric, 0) for metric in self.METRICS), 0
        else:
            sentences = 0

        if evaluating: self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            # Generate batch
            sentence_lens, b = dataset.next_batch(args.batch_size)

            # Prepare feeds
            feeds = {self.is_training: False, self.sentence_lens: sentence_lens,
                     self.charseqs: b[train.FORMS].charseqs, self.charseq_lens: b[train.FORMS].charseq_lens,
                     self.word_ids: b[train.FORMS].word_ids, self.charseq_ids: b[train.FORMS].charseq_ids}
            if args.embeddings:
                embeddings = np.zeros([b[train.EMBEDDINGS].word_ids.shape[0], b[train.EMBEDDINGS].word_ids.shape[1], args.embeddings_size])
                for i in range(embeddings.shape[0]):
                    for j in range(embeddings.shape[1]):
                        if b[train.EMBEDDINGS].word_ids[i, j]:
                            embeddings[i, j] = args.embeddings_data[b[train.EMBEDDINGS].word_ids[i, j] - 1]
                feeds[self.embeddings] = embeddings
            if args.elmo_size:
                feeds[self.elmo] = b[train.ELMOS].word_ids
            if evaluating:
                for factor in args.factors: feeds[self.factors[factor]] = b[train.FACTORS_MAP[factor]].word_ids

            # Prepare targets and run the network
            targets = [self.predictions]
            any_analyses = any(b[train.FACTORS_MAP[factor]].analyses_ids for factor in args.factors)
            if any_analyses: targets.append(self.prediction_probs)
            if evaluating: targets.append(self.update_loss)
            predictions_raw, *other_values = self.session.run(targets, feeds)
            if any_analyses: prediction_probs, *other_values = other_values

            # Use analyses if given
            if any_analyses:
                predictions = dict((factor, predictions_raw[factor].copy()) for factor in predictions_raw)
                for i in range(len(sentence_lens)):
                    for j in range(sentence_lens[i]):
                        analysis_ids = [b[dataset.FACTORS_MAP[factor]].analyses_ids[i][j] for factor in args.factors]
                        if not analysis_ids or len(analysis_ids[0]) == 0: continue

                        known_analysis = any(all(analysis_ids[f][a] != dataset.UNK for f in range(len(args.factors)))
                                             for a in range(len(analysis_ids[0])))
                        if not known_analysis: continue

                        # Compute probabilities of unknown analyses as minimum probability
                        # of a known analysis - 1e-3.
                        analysis_probs = [prediction_probs[factor][i, j] for factor in args.factors]
                        for f in range(len(args.factors)):
                            min_probability = None
                            for analysis in analysis_ids[f]:
                                if analysis != dataset.UNK and (min_probability is None or analysis_probs[f][analysis] - 1e-3 < min_probability):
                                    min_probability = analysis_probs[f][analysis] - 1e-3
                            analysis_probs[f][dataset.UNK] = min_probability
                            analysis_probs[f][dataset.PAD] = min_probability

                        # Choose most likely analysis
                        best_index, best_prob = None, None
                        for index in range(len(analysis_ids[0])):
                            prob = sum(analysis_probs[f][analysis_ids[f][index]] for f in range(len(args.factors)))
                            if best_index is None or prob > best_prob:
                                best_index, best_prob = index, prob
                        for f in range(len(args.factors)):
                            predictions[args.factors[f]][i, j] = analysis_ids[f][best_index] if evaluating else -best_index - 1
            else:
                predictions = predictions_raw

            # Compute metrics or generate output
            if evaluating:
                for name_dict, metric_preds in [("Raw", predictions_raw), ("Dict", predictions)]:
                    for factors in [["Lemmas"], ["Tags"], ["Lemmas", "Tags"]]:
                        metric = "".join(factors) + name_dict
                        if metric in metrics:
                            for i in range(len(sentence_lens)):
                                for j in range(sentence_lens[i]):
                                    metrics[metric] += all(b[dataset.FACTORS_MAP[factor]].word_ids[i, j] == metric_preds[factor][i, j]
                                                           for factor in factors)
                words += sum(sentence_lens)
            else:
                for i in range(len(sentence_lens)):
                    overrides = [None] * dataset.FACTORS
                    for factor in args.factors: overrides[dataset.FACTORS_MAP[factor]] = predictions[factor][i]
                    dataset.write_sentence(output, sentences, overrides)
                    sentences += 1

        if evaluating:
            for metric in metrics:
                metrics[metric] /= words
            return metrics

    def evaluate(self, dataset_name, dataset, args):
        metrics = self.predict(dataset, None, args)
        self.session.run(self.summaries[dataset_name],
                         dict((self.metrics[metric], metrics[metric]) for metric in self.METRICS))
        return metrics


if __name__ == "__main__":
    import argparse
    import datetime
    import json
    import os
    import sys
    import re

    # Fix random seed
    np.random.seed(42)

    command_line = " ".join(sys.argv[1:])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Input data")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    parser.add_argument("--char_dropout", default=0, type=float, help="Character dropout")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--elmo", default=None, type=str, help="External contextualized embeddings to use.")
    parser.add_argument("--embeddings", default=None, type=str, help="External embeddings to use.")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--factors", default="Lemmas,Tags", type=str, help="Factors to predict.")
    parser.add_argument("--factor_layers", default=1, type=int, help="Per-factor layers.")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--lemma_re_strip", default=r"(?<=.)(?:`|_|-[^0-9]).*$", type=str, help="RE suffix to strip from lemma.")
    parser.add_argument("--lemma_rule_min", default=2, type=int, help="Minimum occurences to keep a lemma rule.")
    parser.add_argument("--min_epoch_batches", default=300, type=int, help="Minimum number of batches per epoch.")
    parser.add_argument("--predict", default=None, type=str, help="Predict using the passed model.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=3, type=int, help="RNN layers.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    args = parser.parse_args()

    if args.predict:
        # Load saved options from the model
        with open("{}/options.json".format(args.predict), mode="r") as options_file:
            args = argparse.Namespace(**json.load(options_file))
        parser.parse_args(namespace=args)
    else:
        # Create logdir name
        if args.exp is None:
            args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

        do_not_log = {"exp", "lemma_re_strip", "predict", "threads"}
        args.logdir = "models/{}-{}".format(
            args.exp,
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("[^,]*/", "", value) if type(value) == str else value)
                      for key, value in sorted(vars(args).items()) if key not in do_not_log))
        )
        if not os.path.exists("models"): os.mkdir("models")
        if not os.path.exists(args.logdir): os.mkdir(args.logdir)

        # Dump passed options
        with open("{}/options.json".format(args.logdir), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True)

    # Postprocess args
    args.factors = args.factors.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Load embeddings
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=True) as embeddings_npz:
            args.embeddings_words = embeddings_npz["words"]
            args.embeddings_data = embeddings_npz["embeddings"]
            args.embeddings_size = args.embeddings_data.shape[1]

    if args.predict:
        # Load training dataset maps from the checkpoint
        train = morpho_dataset.MorphoDataset.load_mappings("{}/mappings.pickle".format(args.predict))
        # Load input data
        predict = morpho_dataset.MorphoDataset(args.data, train=train, shuffle_batches=False, elmo=args.elmo)
    else:
        # Load input data
        train = morpho_dataset.MorphoDataset("{}-train.txt".format(args.data),
                                             embeddings=args.embeddings_words if args.embeddings else None,
                                             elmo=re.sub("(?=,|$)", "-train.npz", args.elmo) if args.elmo else None,
                                             lemma_re_strip=args.lemma_re_strip,
                                             lemma_rule_min=args.lemma_rule_min)
        if os.path.exists("{}-dev.txt".format(args.data)):
            dev = morpho_dataset.MorphoDataset("{}-dev.txt".format(args.data), train=train, shuffle_batches=False,
                                               elmo=re.sub("(?=,|$)", "-dev.npz", args.elmo) if args.elmo else None)
        else:
            dev = None

        if os.path.exists("{}-test.txt".format(args.data)):
            test = morpho_dataset.MorphoDataset("{}-test.txt".format(args.data), train=train, shuffle_batches=False,
                                               elmo=re.sub("(?=,|$)", "-test.npz", args.elmo) if args.elmo else None)
        else:
            test = None
    args.elmo_size = train.elmo_size

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args=args,
                      num_words=len(train.factors[train.FORMS].words),
                      num_chars=len(train.factors[train.FORMS].alphabet),
                      factor_words=dict((factor, len(train.factors[train.FACTORS_MAP[factor]].words)) for factor in args.factors),
                      predict_only=args.predict)

    if args.predict:
        network.saver_inference.restore(network.session, "{}/checkpoint-inference".format(args.predict))
        network.predict(predict, sys.stdout, args)

    else:
        log_file = open("{}/log".format(args.logdir), "w")
        for factor in args.factors:
            print("{}: {}".format(factor, len(train.factors[train.FACTORS_MAP[factor]].words)), file=log_file, flush=True)
        print("Tagging with args:", "\n".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items())
                                               if key not in ["embeddings_data", "embeddings_words"])), flush=True)

        for i, (epochs, learning_rate) in enumerate(args.epochs):
            for epoch in range(epochs):
                network.train_epoch(train, learning_rate, args)

                if dev:
                    metrics = network.evaluate("dev", dev, args)
                    metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
                    for f in [sys.stderr, log_file]:
                        print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f, flush=True)

        network.saver_inference.save(network.session, "{}/checkpoint-inference".format(args.logdir), write_meta_graph=False)
        train.save_mappings("{}/mappings.pickle".format(args.logdir))

        if test:
            metrics = network.evaluate("test", test, args)
            metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
            for f in [sys.stderr, log_file]:
                print("Test, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f, flush=True)
