"""Retained Light consumers use the owning modules for shared rules."""
from aka.legacy.execution.configuration import DEFAULT_QUEUE_WAIT_GRACE, queue_wait_grace
from aka.legacy.evaluation.metadata import distributed_evaluation_world_size, evaluator_num_gpus
