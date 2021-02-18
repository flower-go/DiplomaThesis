from text_classification_dataset import TextClassificationDataset
import tensorflow_datasets as tfds
import pandas as pd
import os
import zipfile

# loading csfd and mall datset from: https://github.com/kysely/sentiment-analysis-czech/blob/sentence-level/Sentiment%20Analysis%20in%20Czech.ipynb

class SentimentDataset():


    def __init__(self, tokenizer):

        self.labels = {'n': 1, '0': 0, 'p': 2, 'b': 'BIP'}
        self.target_labels = [self.labels['n'], self.labels['0'], self.labels['p']]
        #self.max_sentence_length = 30  # no. of words %TODO is it true?
        self.tokenizer = tokenizer


    def get_dataset(self, dataset_name, path=None, debug=False):
        if dataset_name == "facebook":
            if self.tokenizer is not None:
                return TextClassificationDataset(path + "/" + "czech_facebook", tokenizer=self.tokenizer.encode)
            else:
                return self._load_facebook(path + "/" + "czech_facebook")
        if dataset_name == "imdb":
            return self._return_imdb(self.tokenizer)
        if dataset_name == "csfd":
            path = path + "/" + dataset_name
            if debug:
                return self.load_data(path, True)
            else:
                return self.load_data(path)
        if dataset_name == "mall":
            path = path + "/" + dataset_name + "cz"
            return self.load_data(path)


    def _load_facebook(self, path):
        tokens= []
        labels=[]
        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0].split("/")[-1], dataset),
                                   "r") as dataset_file:
                    for line in dataset_file:
                        line = line.decode("utf-8").rstrip("\r\n")
                        label, text = line.split("\t", maxsplit=1)

                        tokens.append(text)
                        labels.append(label)
        return pd.DataFrame({"Post": tokens, "Sentiment": labels})

    def _return_imdb(self, tokenizer):


        train_data, train_labels = tfds.load(name="imdb_reviews", split="train",
                                      batch_size=-1, as_supervised=True)

        train_examples = tfds.as_numpy(train_data)
        train_examples = self._imdb_covertion(train_examples, tokenizer)


        return train_examples, train_labels

    def _imdb_covertion(self,data,tokenizer):
        for i in range(len(data)):

            if len(data[i]) > 512:
                data[i] = data[i][0:512]

            data[i] = tokenizer.encode(data[i].decode('latin1'))
        return data

    def load_gold_data(self,directory, filter_out):
        '''
        Loads a dataset with separate contents and labels. Maps labels to our format.
        Filters out any samples that have a label equal to the second argument.

        Returns a new DataFrame.
        '''
        return pd \
            .concat([
            pd.read_csv('data/{}/gold-posts.txt'.format(directory), sep='\n', header=None, names=['Post']),
            pd.read_csv('data/{}/gold-labels.txt'.format(directory), sep=' ', header=None, names=['Sentiment']).iloc[:,
            0].map(self.labels)
        ], axis=1) \
            .query('Sentiment != @filter_out') \
            .reset_index(drop=True)

    def load_data(self,directory, debug=False):
        '''
        Loads a dataset whose samples are split to individual files/per class.

        Returns a new DataFrame.
        '''
        if debug:
            return pd \
                .concat([
                pd.read_csv('{}/positive-small.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=self.labels['p']),
                pd.read_csv('{}/neutral-small.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=self.labels['0']),
                pd.read_csv('{}/negative-small.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=self.labels['n'])
            ], axis=0) \
                .reset_index(drop=True)
        else:
            return pd \
                .concat([
                pd.read_csv('{}/positive.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=self.labels['p']),
                pd.read_csv('{}/neutral.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=self.labels['0']),
                pd.read_csv('{}/negative.txt'.format(directory), sep='\n', header=None, names=['Post']).assign(
                    Sentiment=self.labels['n'])
            ], axis=0) \
                .reset_index(drop=True)

