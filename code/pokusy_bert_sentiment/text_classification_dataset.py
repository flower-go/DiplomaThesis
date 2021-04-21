import os
import sys
import urllib.request
import zipfile

import numpy as np
import pickle

# Loads a text classification dataset in a vertical format.
#
# During the construction a `tokenizer` callable taking a string
# and returning a list/np.ndarray of integers must be given.
class TextClassificationDataset:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    class Dataset:
        LABELS = None # Will be filled during Dataset construction

        def __init__(self, data_file, tokenizer, train=None, shuffle_batches=True, seed=42, from_array=False):
            # Create factors
            self._data = {
                "tokens": [],
                "labels": [],
            }
            self._label_map = train._label_map if train else {}
            self.LABELS = train.LABELS if train else []

            if not from_array:

                for line in data_file:
                    line = line.decode("utf-8").rstrip("\r\n")
                    label, text = line.split("\t", maxsplit=1)

                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    encoded = tokenizer(text)
                    if type(encoded) is dict:
                        encoded = encoded["input_ids"]
                    self._data["tokens"].append(encoded)
                    self._data["labels"].append(label)
            else:
                for i,row in data_file.iterrows():

                    #TODO vyresit label_map

                    text = row["Post"].rstrip("\r\n")[0:512]
                    label = row["Sentiment"]

                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)
                    encoded = tokenizer(text)
                    if type(encoded) is dict:
                        encoded = encoded["input_ids"]
                    self._data["tokens"].append(encoded)
                    self._data["labels"].append(label)


            self._size = len(self._data["tokens"])
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            print(self._size)
            print( len(self._data["tokens"]))
            data_tokens = self._data["tokens"]
            data_labels = self._data["labels"]
            print(data_tokens)
            print(data_labels)

            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                max_sentence_len = max(len(data_tokens[i]) for i in batch_perm)

                tokens = np.zeros([batch_size, max_sentence_len], np.int32)
                labels = np.zeros([batch_size], np.int32)
                for i in range(batch_size):
                    print("batch size " + str(batch_size))
                    print("max" + str(max_sentence_len))
                    print("i" + str(i))
                    print("len " + str(len(data_tokens[batch_perm[i]])))
                    tokens[i, :len(data_tokens[batch_perm[i]])] = data_tokens[batch_perm[i]]
                    labels[i] = data_labels[batch_perm[i]]

                yield tokens, labels

        def append_data(self,tokens,labels):
            self._size = self._size + len(tokens)
            self._data["tokens"].extend(tokens)
            self._data["labels"].extend(labels)


    def __init__(self, dataset=None, tokenizer=None):
        """Create the dataset of the given name.

        The `tokenizer` should be a callable taking a string and returning
        a list/np.ndarray of integers.
        """
        if dataset != None:

            path = "{}.zip".format(dataset)
            print(path)
            if not os.path.exists(path):
                print("Downloading dataset {}...".format(dataset), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)
            if tokenizer is not None:
                with zipfile.ZipFile(path, "r") as zip_file:
                    for dataset in ["train", "dev", "test"]:
                        with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0].split("/")[-1], dataset), "r") as dataset_file:
                            setattr(self, dataset, self.Dataset(dataset_file, tokenizer,
                                                            train=self.train if dataset != "train" else None,
                                                            shuffle_batches=dataset == "train"))


    def from_array(self, data, tokenizer):
        for i,dataset in enumerate(["train", "dev", "test"]):
            print("dataset")
            setattr(self, dataset, self.Dataset(data[i], tokenizer,
                                                    train=self.train if dataset != "train" else None,
                                                    shuffle_batches=dataset == "train", from_array=True))

        return self


    def append_dataset(self,dataset):
        for d in ["train", "dev", "test"]:
            data_orig = getattr(self, d)
            data_new = getattr(dataset,d)
            data_orig.append_data(data_new.data["tokens"], data_new.data["labels"]) #TODO nekotrnoluju stejne labels


