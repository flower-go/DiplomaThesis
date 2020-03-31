#!/usr/bin/env python3
import collections
import json

import transformers
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import morpho_dataset
import pickle


class Network:
    def __init__(self, args, num_words, num_chars, factor_words):

        self.factors = args.factors
        self.factor_words = factor_words

        word_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        charseq_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        if args.embeddings:
            embeddings = tf.keras.layers.Input(shape=[None, args.embeddings_size], dtype=tf.float32)
        if args.elmo_size:
            elmo = tf.keras(shape=[None, args.elmo_size], dtype=tf.float32)
        if args.bert:
            bert_embeddings = tf.keras.layers.Input(shape=[None, args.bert_size], dtype=tf.float32)

        # INPUTS - create all embeddings
        inputs = []
        if args.we_dim:
            inputs.append(tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)(word_ids))

        cle = tf.keras.layers.Embedding(num_chars, args.cle_dim, mask_zero=True)(charseqs)
        cle = tf.keras.layers.Dropout(rate=args.dropout)(cle)
        cle = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim), merge_mode="concat")(cle)
        # cle = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([cle, charseq_ids])
        cle = tf.gather(1 * cle, charseq_ids)

        # If CLE dim is half WE dim, we add them together, which gives
        # better results; otherwise we concatenate CLE and WE.
        if 2 * args.cle_dim == args.we_dim:
            inputs[-1] = tf.keras.layers.Add()([inputs[-1], cle])
        else:
            inputs.append(cle)

        # Pretrained embeddings
        if args.embeddings:
            inputs.append(tf.keras.layers.Dropout(args.word_dropout, noise_shape=[None, None, 1])(embeddings))

        # Contextualized embeddings
        if args.elmo_size:
            inputs.append(elmo)

        # Bert embeddings
        if args.bert:
            inputs.append(tf.keras.layers.Dropout(args.word_dropout, noise_shape=[None, None, 1])(bert_embeddings))

        if len(inputs) > 1:
            hidden = tf.keras.layers.Concatenate()(inputs)
        else:
            hidden = inputs[0]

        # RNN cells

        hidden = tf.keras.layers.Dropout(rate=args.dropout)(hidden)

        for i in range(args.rnn_layers):
            previous = hidden
            rnn_layer = getattr(tf.keras.layers, args.rnn_cell)(args.rnn_cell_dim, return_sequences=True)
            hidden = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")(hidden)
            hidden = tf.keras.layers.Dropout(rate=args.dropout)(hidden)
            if i:
                hidden = tf.keras.layers.Add()([previous, hidden])

        # tagger
        outputs = []
        for factor in args.factors:
            factor_layer = hidden
            for _ in range(args.factor_layers):
                factor_layer = tf.keras.layers.Add()([factor_layer, tf.keras.layers.Dropout(rate=args.dropout)(
                    tf.keras.layers.Dense(args.rnn_cell_dim, activation=tf.nn.tanh)(factor_layer))])
            if factor == "Lemmas":
                factor_layer = tf.keras.layers.Concatenate()([factor_layer, cle])
            outputs.append(tf.keras.layers.Dense(factor_words[factor], activation=tf.nn.softmax)(factor_layer))

        inp = [word_ids, charseq_ids, charseqs]
        if (args.embeddings):
            inp.append(embeddings)
        if args.bert:
            inp.append(bert_embeddings)

        self.model = tf.keras.Model(inputs=inp, outputs=outputs)
        self._optimizer = tfa.optimizers.LazyAdam(beta_2=args.beta_2)

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
        # tags_mask = tf.not_equal(factors[0], 0)
        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
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
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
            self._metrics["loss"](loss)
            for i in range(len(self.factors)):
                self._metrics[self.factors[i] + "Raw"](factors[i], probabilities[i], probabilities[i]._keras_mask)
            for name, metric in self._metrics.items():
                tf.summary.scalar("train/{}".format(name), metric.result())

    def train_epoch(self, dataset, args, learning_rate):
        self._optimizer.learning_rate = learning_rate
        while not train.epoch_finished():
            _, batch = dataset.next_batch(args.batch_size)
            factors = []
            for f in self.factors:
                words = batch[dataset.FACTORS_MAP[f]].word_ids
                factors.append(words)
            inp = [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs]
            print('train epoch')
            if args.embeddings:
                embeddings = np.zeros([batch[train.EMBEDDINGS].word_ids.shape[0],
                                       batch[train.EMBEDDINGS].word_ids.shape[1],
                                       args.embeddings_size])
                for i in range(embeddings.shape[0]):
                    for j in range(embeddings.shape[1]):
                        if batch[train.EMBEDDINGS].word_ids[i, j]:
                            embeddings[i, j] = args.embeddings_data[batch[train.EMBEDDINGS].word_ids[i, j] - 1]
                inp.append(embeddings)

            if args.bert:
                bert_embeddings = np.zeros([batch[train.BERT].word_ids.shape[0],
                                       batch[train.BERT].word_ids.shape[1],
                                       args.bert_size])
                for i in range(bert_embeddings.shape[0]):
                    for j in range(bert_embeddings.shape[1]):
                        if batch[train.BERT].word_ids[i, j]:
                            bert_embeddings[i, j] = train.bert_embeddings[batch[train.BERT].word_ids[i, j] - 1]

                inp.append(bert_embeddings)

            self.train_batch(inp, factors)

    @tf.function(experimental_relax_shapes=True)
    def evaluate_batch(self, inputs, factors):
        probabilities = self.model(inputs, training=False)
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
                embeddings = np.zeros([batch[dataset.EMBEDDINGS].word_ids.shape[0],
                                       batch[dataset.EMBEDDINGS].word_ids.shape[1],
                                       args.embeddings_size])
                for i in range(embeddings.shape[0]):
                    for j in range(embeddings.shape[1]):
                        if batch[dataset.EMBEDDINGS].word_ids[i, j]:
                            embeddings[i, j] = args.embeddings_data[batch[dataset.EMBEDDINGS].word_ids[i, j] - 1]
                inp.append(embeddings)

            if args.bert:
                bert_embeddings = np.zeros([batch[dataset.BERT].word_ids.shape[0],
                                       batch[dataset.BERT].word_ids.shape[1],
                                       args.bert_size])
                for i in range(bert_embeddings.shape[0]):
                    for j in range(bert_embeddings.shape[1]):
                        if batch[dataset.BERT].word_ids[i, j]:
                            bert_embeddings[i, j] = args.bert_data[batch[dataset.BERT].word_ids[i, j] - 1]
                inp.append(bert_embeddings)

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

                            #TODO spatny shape, probabilities f analysis je vektor
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
    parser.add_argument("--elmo", default=None, type=str, help="External contextualized embeddings to use.")
    parser.add_argument("--embeddings", default=None, type=str, help="External embeddings to use.")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--factors", default="Tags", type=str, help="Factors to predict.")
    parser.add_argument("--factor_layers", default=1, type=int, help="Per-factor layers.")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
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
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="Bert model")
    args = parser.parse_args()
    args.debug_mode = args.debug_mode == 1

    #TODO vyřešit
    #tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    #tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    #tf.config.set_soft_device_placement(True)

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
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key),
                                     re.sub("[^,]*/", "", value) if type(value) == str else value)
                      for key, value in sorted(vars(args).items()) if key not in do_not_log))
        )
        if not os.path.exists("models"): os.mkdir("models")
        if not os.path.exists(args.logdir): os.mkdir(args.logdir)

        # Dump passed options
        with open("{}/options.json".format(args.logdir), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True)

    # TODO write summaries using logdir

    # Postprocess args
    args.factors = args.factors.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in
                   (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Load embeddings
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=True) as embeddings_npz:
            args.embeddings_words = embeddings_npz["words"]
            args.embeddings_data = embeddings_npz["embeddings"]
            args.embeddings_size = args.embeddings_data.shape[1]


    # args.compute_bert = False
    # if args.bert:
    #     bert_path = args.bert + ".pickle"
    #     if os.path.exists(bert_path):
    #         print("cesta existuje")
    #         args.compute_bert = False
    #         bert_pickle = np.load(bert_path, allow_pickle=True)
    #         args.bert_words = bert_pickle[0]
    #         print("size of berts")
    #         print(len(args.bert_words))
    #         args.bert_data = bert_pickle[1]
    #         args.bert_size = len(args.bert_data[0])
    #     else:
    #         args.bert_words = None
    #         args.compute_bert = True


    if args.predict:
        # Load training dataset maps from the checkpoint
        train = morpho_dataset.MorphoDataset.load_mappings("{}/mappings.pickle".format(args.predict))
        # Load input data
        predict = morpho_dataset.MorphoDataset(args.data, train=train, shuffle_batches=False, elmo=args.elmo)
    else:
        # Load input data
        if args.debug_mode:
            print("DEBUG MODE")
            train_data_path = "{}-train-small.txt".format(args.data)
            dev_data_path = "{}-dev-small.txt".format(args.data)
        else:
            train_data_path = "{}-train.txt".format(args.data)
            dev_data_path = "{}-dev.txt".format(args.data)


        train = morpho_dataset.MorphoDataset(train_data_path,
                                             embeddings=args.embeddings_words if args.embeddings else None,
                                             elmo=re.sub("(?=,|$)", "-train.npz", args.elmo) if args.elmo else None,
                                             bert=args.bert if args.bert else None,
                                             lemma_re_strip=args.lemma_re_strip,
                                             lemma_rule_min=args.lemma_rule_min)

        if os.path.exists(dev_data_path):
            dev = morpho_dataset.MorphoDataset(dev_data_path, train=train, shuffle_batches=False,
                                               bert=args.bert if args.compute_bert else None,
                                               elmo=re.sub("(?=,|$)", "-dev.npz", args.elmo) if args.elmo else None)
        else:
            dev = None


        if os.path.exists("{}-test.txt".format(args.data)):
            test_data_path = "{}-test.txt".format(args.data)
            test = morpho_dataset.MorphoDataset("{}-test.txt".format(args.data), train=train, shuffle_batches=False,
                                                elmo=re.sub("(?=,|$)", "-test.npz", args.elmo) if args.elmo else None,
                                                bert=args.bert if args.bert else None
                                                )
        else:
            test = None
        # test_data_path = "djfha"
        # test = None
    args.elmo_size = train.elmo_size
    args.bert_size = len(train.bert_embeddings[0])

    # if args.compute_bert:
    #     args.bert_words = None
    #     args.bert_data = None
    #     #TODO věci v listu by měly být unikátní
    #     for name in [train_data_path, dev_data_path, test_data_path]:
    #         name = args.bert + "_" + "_".join(name.split("-")[-2:])
    #         name = name + ".pickle"
    #         print(name)
    #         if os.path.exists(name):
    #             bert_pickle = np.load(name, allow_pickle=True)
    #             if args.bert_words is not None:
    #                 args.bert_words = np.concatenate([args.bert_words,bert_pickle[0]])
    #                 args.bert_data = np.concatenate([args.bert_data,bert_pickle[1]])
    #             else:
    #                 args.bert_words = bert_pickle[0]
    #                 args.bert_data = bert_pickle[1]
    #
    #     for_save = [args.bert_words, args.bert_data]
    #     with open(args.bert + '.pickle', 'wb') as handle:
    #         pickle.dump(for_save, handle)
    #
    #
    # #TODO stejne potrebuju ty predchozi promenne - bert_words, bert_data - mam ulozene, bert_size zjistim
    # args.bert_size = len(args.bert_data[0])

    # Construct the network
    network = Network(args=args,
                      num_words=len(train.factors[train.FORMS].words),
                      num_chars=len(train.factors[train.FORMS].alphabet),
                      factor_words=dict(
                          (factor, len(train.factors[train.FACTORS_MAP[factor]].words)) for factor in args.factors))

    # TODO ukladani nefunguje
    # TODO nemame predikci
    if args.predict:
        network.saver_inference.restore(network.session, "{}/checkpoint-inference".format(args.predict))
        network.predict(predict, sys.stdout, args)

    else:
        log_file = open("{}/log".format(args.logdir), "w")
        for factor in args.factors:
            print("{}: {}".format(factor, len(train.factors[train.FACTORS_MAP[factor]].words)), file=log_file,
                  flush=True)
        print("Tagging with args:", "\n".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items())
                                               if key not in ["embeddings_data", "embeddings_words"])), flush=True)

        for i, (epochs, learning_rate) in enumerate(args.epochs):
            for epoch in range(epochs):
                network.train_epoch(train, args, learning_rate)

                if dev:
                    metrics = network.evaluate(dev, "dev", args)
                    metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
                    for f in [sys.stderr, log_file]:
                        print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f,
                              flush=True)

        # network.saver_inference.save(network.session, "{}/checkpoint-inference".format(args.logdir), write_meta_graph=False)
        train.save_mappings("{}/mappings.pickle".format(args.logdir))

        if test:
            metrics = network.evaluate(test, "test", args)
            metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric]) for metric in metrics))
            for f in [sys.stderr, log_file]:
                print("Test, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=f, flush=True)
