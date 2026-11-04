"""Durable elapsed budgets for sequential Evaluation steps."""
import time
from threading import Event


class BudgetCancellation:
    def __init__(self, deadline, *parents):
        self.deadline = deadline
        self.parents = tuple(parent for parent in parents if parent is not None)

    def is_set(self):
        return time.time() >= self.deadline or any(parent.is_set() for parent in self.parents)

    def wait(self, timeout=None):
        end = self.deadline if timeout is None else min(self.deadline, time.time() + timeout)
        while not self.is_set():
            remaining = end - time.time()
            if remaining <= 0:
                return False
            Event().wait(min(remaining, 0.1))
        return True
