import morpho_dataset
import os
import numpy as np

class SimpleDataset(morpho_dataset):

    def __init__(self, debug, data,model):
        train,dev,test = self.return_simple_data(debug,data,model)
        #TODO zbyle datasety
        self.train_encodings = train.bert_subwords
        train_tag_labels = train._factors[train.TAGS].word_ids
        self.train_segments = train.bert_segments
        self.train_tag_labels = self.encode_tags(train_tag_labels, self.train_segments)

        #TODO mozna factor_words

        self.NUM_TAGS = len(train.factors[0].words_map)


    def encode_tags(self, tags, segments):
        labels = tags
        encoded_labels = []
        for doc_labels, s in zip(labels, segments):
            # create an empty array of -100, -100 is used in the place of other-than-first subtokens of word
            doc_enc_labels = np.ones(len(s), dtype=int) * -100
            first_indices = np.nonzero(np.r_[1, np.diff(s)])

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

    def return_simple_data(self, debug, data, model):
        data_paths = self._prepare_paths(data, debug)

        train = morpho_dataset.MorphoDataset(data_paths[0],
                                             embeddings=None,
                                             bert=model,
                                             lemma_re_strip=r"(?<=.)(?:`|_|-[^0-9]).*$",
                                             lemma_rule_min=2, simple=True)

        if os.path.exists(data_paths[1]):
            dev = morpho_dataset.MorphoDataset(data_paths[1], train=train, shuffle_batches=False,
                                               bert=model, simple=True
                                               )
        else:
            dev = None

        if os.path.exists(data_paths[2]):
            test = morpho_dataset.MorphoDataset(data_paths[2], train=train, shuffle_batches=False,
                                                bert=model, simple=True
                                                )
        else:
            test = None

        return train,dev,test
