import morpho_dataset
import os
import numpy as np
import transformers

class BertModel:
    def __init__(self, name):
        self.name = name
        self.config = transformers.BertConfig.from_pretrained(name)
        self.config.output_hidden_states = True
        self.tokenizer = transformers.BertTokenizer.from_pretrained(name)
        self.model = transformers.TFBertModel.from_pretrained(name,
                                                              config=self.config)
        self.embeddings_only = False

class SimpleDataset():
    class FactorBatch:
        def __init__(self, word_ids, charseq_ids=None, charseqs=None, charseq_lens=None, analyses_ids=None):
            self.word_ids = word_ids
            self.charseq_ids = charseq_ids
            self.charseqs = charseqs
            self.charseq_lens = charseq_lens
            self.analyses_ids = analyses_ids

    def __init__(self, debug, data, name,model, train=None):
        model_bert = BertModel(model)
        self.data = self.return_simple_data(debug,data,model_bert, name, train=None)
        self._sentence_lens = np.array([len(s)+2 for s in self.data.bert_segments])
        self._permutation = np.random.permutation(len(self._sentence_lens)) if self.data._shuffle_batches else np.arange(
            len(self._sentence_lens))

        self.data._factors[self.data.TAGS].word_ids = self.encode_tags(self.data._factors[self.data.TAGS].word_ids,
                                                                          self.data.bert_segments)


        self.NUM_TAGS = len(self.data.factors[0].words_map)


    def encode_tags(self, tags, segments):
        labels = tags
        encoded_labels = []
        for doc_labels, s in zip(labels, segments):
            # create an empty array of -100, -100 is used in the place of other-than-first subtokens of word
            doc_enc_labels = np.zeros(len(s)+2, dtype=int)
            first_indices = np.nonzero(np.r_[1, np.diff(s)])[0]
            first_indices = first_indices + 1

            doc_enc_labels[first_indices] = doc_labels
            #doc_enc_labels = doc_enc_labels + 1
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    def _prepare_paths(self,data, debug = False):
        # Load input data
        data_paths = [None] * 3
        if debug:
            print("DEBUG MODE")
            data_paths[0] = "{}-train-small.txt".format(data)
            data_paths[1] = "{}-dev-small.txt".format(data)
            data_paths[2] = "{}-test-small.txt".format(data)
        else:
            data_paths[0] = "{}-train.txt".format(data)
            data_paths[1] = "{}-dev.txt".format(data)
            data_paths[2] = "{}-test.txt".format(data)
        return data_paths

    def return_simple_data(self, debug, data, model, name,train):
        data_paths = self._prepare_paths(data, debug)

        if name == "train":
            train = morpho_dataset.MorphoDataset(data_paths[0],
                                                 embeddings=None,
                                                 bert=model,
                                                 lemma_re_strip=r"(?<=.)(?:`|_|-[^0-9]).*$",
                                                 lemma_rule_min=2, simple=True)

        if name == "dev":
            if os.path.exists(data_paths[1]):
                dev = morpho_dataset.MorphoDataset(data_paths[1], train=train, shuffle_batches=False,
                                                   bert=model, simple=True
                                                   )
            else:
                dev = None
            return dev

        if name == "test":
            if os.path.exists(data_paths[2]):
                test = morpho_dataset.MorphoDataset(data_paths[2], train=train, shuffle_batches=False,
                                                    bert=model, simple=True
                                                    )
            else:
                test = None
            return test

        return train

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        # General data

        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        factors = []
        factor = self.data._factors[0]
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        for i in range(batch_size):
            factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = self.data.bert_subwords[batch_perm[i]]

        factor = self.data._factors[2] #jen tags
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        for i in range(batch_size):
            factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = factor.word_ids[batch_perm[i]]

        return batch_sentence_lens, factors

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self.data._shuffle_batches else np.arange(
                len(self._sentence_lens))
            return True
        return False
