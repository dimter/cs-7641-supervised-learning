from abc import ABC, abstractmethod
import pandas as pd
import logging
import sklearn.preprocessing
import sklearn.model_selection


class Dataset(ABC):
    def __init__(
            self,
            name: str,
            filename: str,
            filetype: str,
            delimiter: str = None,
            random_state: int = 1):
        self.name = name
        self.filename = filename
        self.filetype = filetype
        self.delimiter = delimiter
        self.random_state = random_state
        self.dataframe = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def read_csv(self):
        self.logger.info(f'Reading dataset {self.name} as csv')
        self.dataframe = pd.read_csv(self.filename, delimiter=self.delimiter)

    def read(self):
        if self.filetype == 'csv':
            self.read_csv()

    def split_x_y(self):
        self.x = self.dataframe.iloc[:, :-1]
        self.y = self.dataframe.iloc[:, -1].values

    def scale_x(self):
        self.x = x = sklearn.preprocessing.scale(self.x)

    def prepare_x(self):
        pass

    def prepare_y(self):
        pass

    def check(self):
        if self.x.isnull().values.any():
            self.logger.warning('Dataset has null values!')
        if (self.x.values < 0).any():
            self.logger.warning('Dataset contains negative values!')
        if (self.y < 0).any():
            self.logger.warning('Dataset contains negative values!')

    def split_train_test(self, test_size: int = 0.8):
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            self.x, self.y, test_size=test_size, random_state=self.random_state)

    def load(self, test_size: int = 0.8):
        self.read()
        self.split_x_y()
        self.prepare_x()
        self.prepare_y()
        self.check()
        self.scale_x()
        self.logger.info(
            f'Dataset contains {self.dataframe.shape[0]} rows and {self.x.shape[1]} features')


class WineDataset(Dataset):
    def __init__(self, cutoff_threshold=6):
        self.cutoff_threshold = cutoff_threshold
        super().__init__(name='wine', filename='data/winequality-white.csv', filetype='csv', delimiter=';')

    def prepare_y(self):
        self.y = self.y.astype(int)
        self.y[self.y < self.cutoff_threshold] = 0
        self.y[self.y >= self.cutoff_threshold] = 1
        self.logger.info(
            f'Number of 0 samples: {len(self.y[self.y == 0])} ({len(self.y[self.y == 0]) / len(self.y)})')
        self.logger.info(
            f'Number of 1 samples: {len(self.y[self.y == 1])}({len(self.y[self.y == 1]) / len(self.y)})')


class BreastCancerDataset(Dataset):
    def __init__(self, cutoff_threshold=6):
        self.cutoff_threshold = cutoff_threshold
        super().__init__(name='breast cancer', filename='data/breast-cancer.csv', filetype='csv', delimiter=',')

    def split_x_y(self):
        self.x = self.dataframe.iloc[:, 2:-1]
        self.y = self.dataframe.iloc[:, 1].values

    def prepare_y(self):
        self.y[self.y == 'M'] = 1
        self.y[self.y == 'B'] = 0
        self.y = self.y.astype(int)
        self.logger.info(
            f'Number of 0 samples: {len(self.y[self.y == 0])} ({len(self.y[self.y == 0]) / len(self.y)})')
        self.logger.info(
            f'Number of 1 samples: {len(self.y[self.y == 1])}({len(self.y[self.y == 1]) / len(self.y)})')
