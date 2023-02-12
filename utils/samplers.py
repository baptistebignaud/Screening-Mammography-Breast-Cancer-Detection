from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

# cf. https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7 answer of Reuben Feinman
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        assert len(y.shape) == 1, "label array must be 1D"
        n_batches = int(len(y) / batch_size)
        self.batch_size = batch_size
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            torch.randint(0, int(1e8), size=()).item()
        for _, indices in self.skf.split(self.X, self.y):
            yield indices

    def __len__(self):
        return len(self.y) // self.batch_size


# TODO code ImbalancedDatasetSampler https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
class ImbalancedDatasetSampler:
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size

        # distribution of classes in the dataset
        label_to_count = self.labels.value_counts()
        weights = 1.0 / label_to_count[self.labels]

        self.weights = torch.DoubleTensor(weights.to_list())
        self.num_samples = len(self.labels)
        self.indices = [
            # self.indices[i]
            i.item()
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        ]

    def __iter__(self):
        # indices = [
        #     # self.indices[i]
        #     i.item()
        #     for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        # ]
        for indices in np.array_split(self.indices, self.__len__()):
            yield indices

    def __len__(self):
        return len(self.labels) // self.batch_size
