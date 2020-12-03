from text_classification_dataset import TextClassificationDataset

class SentimentDataset():
    def __init__(self, dataset_name, tokenizer):
        if dataset_name == "facebook":
            self.data = TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)
        if dataset_name == "imdb":
            ...

