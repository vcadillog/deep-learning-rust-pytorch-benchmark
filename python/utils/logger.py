from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Logger():

    def __init__(self) -> None:
        self.data = []
        self.labels = []

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def set_names(self, labels: List) -> None:
        self.labels = labels

    def set_data(self, data: List) -> None:
        self.data = data

    def get(self) -> None:
        self._strategy.log(self.data, self.labels)


class Strategy(ABC):
    @abstractmethod
    def log(self, data: List, labels: List):
        pass


class LogCSV(Strategy):
    def __init__(self) -> None:
        self.df = pd.DataFrame()

    def clean_data(self):
        self.df = pd.DataFrame()

    def save_csv(self, labels: str):
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(labels+'.csv')

    def log(self, data: List, labels: List) -> None:
        tmp = pd.DataFrame([data], columns=labels)
        self.df = pd.concat([self.df, tmp])


class LogConsole(Strategy):
    def log(self, data: List, labels: List) -> None:
        buffer = ''
        if len(data) != len(labels):
            print("The number of labels doesn't match with the data")
            size = min(len(data), len(labels))
        else:
            size = len(data)
        for (name, value) in zip(labels, data):
            buffer += name + ': ' + str(value)
            size -= 1
            if size > 0:
                buffer += ' || '
        print(buffer)
