from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError()

    @abstractmethod
    def update(self, state, action, reward, new_state, done):
        raise NotImplementedError()