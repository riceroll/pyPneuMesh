from abc import ABC, abstractmethod

from utils.truss import Truss


# each objective takes in an input
class Objective(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass
