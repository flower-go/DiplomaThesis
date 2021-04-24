import morpho_dataset
import os
import numpy as np
import math
import transformers
import tensorflow as tf

class SimpleDataset():
    class FactorBatch:
        def __init__(self, word_ids, charseq_ids=None, charseqs=None, charseq_lens=None, analyses_ids=None):
            self.word_ids = word_ids
            self.charseq_ids = charseq_ids
            self.charseqs = charseqs
            self.charseq_lens = charseq_lens
            self.analyses_ids = analyses_ids

    def __init__(self, debug, data, name,model, train=None):
        model_bert = model
        self.data = self.return_simple_data(debug,data,model_bert, name, train)
        self._sentence_lens = np.array([len(s)+2 for s in self.data.bert_segments])
        self._permutation = np.random.permutation(len(self._sentence_lens)) if self.data._shuffle_batches else np.arange(
            len(self._sentence_lens))


        self.data._factors[self.data.TAGS].word_ids = self.encode_tags(self.data._factors[self.data.TAGS].word_ids,
                                                                       self.data.bert_segments)
        self.data._factors[self.data.LEMMAS].word_ids = self.encode_tags(self.data._factors[self.data.LEMMAS].word_ids,
                                                                       self.data.bert_segments)

        #TODO charseq_ids

        self.data._factors[self.data.FORMS].charseq_ids = self.encode_tags(self.data._factors[self.data.FORMS].charseq_ids, self.data.bert_segments)
        self.NUM_TAGS = len(self.data.factors[self.data.TAGS].words_map)
        self.NUM_LEMMAS = len(self.data.factors[self.data.LEMMAS].words_map)
        self.num_chars = len(self.data.factors[self.data.FORMS].alphabet)




    def encode_tags(self, tags, segments):
        labels = tags
        encoded_labels = []
        for doc_labels, s in zip(labels, segments):
            # create an empty array of 0 is used in the place of other-than-first subtokens of word
            doc_enc_labels = np.zeros(len(s)+2, dtype=int)
            #print("encoding magic")
            #print(np.diff(s))
            #print(np.r_[1, np.diff(s)])
            #print(np.nonzero(np.r_[1, np.diff(s)]))
            first_indices = np.nonzero(np.r_[1, np.diff(s)])[0]
            first_indices = first_indices + 1
            #print(doc_labels)
            doc_enc_labels[first_indices] = doc_labels
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

    def next_batch(self, batch_size,train=0):
       
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        if train > 0:
            num_tokens = sum(batch_sentence_lens) - (len(batch_sentence_lens)*2)
            print("num tokens " + str(num_tokens))
            print("sentence lens " + str(batch_sentence_lens))
            num_del = math.floor(train*num_tokens)
            a = np.zeros(num_tokens, dtype=int)
            a[:num_del] = 1
            np.random.shuffle(a)
            s = 0
            print(str(a))
        factors = []
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        for i in range(batch_size):
            print("delka vety" + str(batch_sentence_lens[i]))
            factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = self.data.bert_subwords[batch_perm[i]]
            print(factors[-1].word_ids[i,0:batch_sentence_lens[i]])
            print(self.data.bert_subwords[batch_perm[i]])
            if train > 0:
                start = s
                print("start " + str(start))
                end = s + batch_sentence_lens[i] - 1
                print("end " + str(end))
                s = s + batch_sentence_lens[i]
                indices = np.nonzero(a[start:end])
                if len(indices) >0 :
                    print(indices)
                    indices = np.array(indices) + 1
                    factors[-1].word_ids[i,indices] = self.data.tokenizer.mask_token_id

        factor = self.data._factors[self.data.LEMMAS]
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        for i in range(batch_size):
            factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = factor.word_ids[batch_perm[i]]

        factor = self.data._factors[self.data.TAGS]
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        for i in range(batch_size):
            factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = factor.word_ids[batch_perm[i]]

        tf.print(factor.word_ids[batch_perm[i]], summarize=-1)
        # Character-level data
        for f, factor in enumerate(self.data._factors):
            if not factor.characters: continue

            factors[f].charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
            charseqs_map = {}
            charseqs = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(factor.charseqs[charseq_id])
                    factors[f].charseq_ids[i, j] = charseqs_map[charseq_id]

            factors[f].charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
            factors[f].charseqs = np.zeros([len(charseqs), np.max(factors[f].charseq_lens)], np.int32)
            for i in range(len(charseqs)):
                factors[f].charseqs[i, 0:len(charseqs[i])] = charseqs[i]

        return batch_sentence_lens, factors

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self.data._shuffle_batches else np.arange(
                len(self._sentence_lens))
            return True
        return False
