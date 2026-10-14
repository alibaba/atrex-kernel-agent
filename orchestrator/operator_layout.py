"""Compatibility imports for the task-owned operator layout implementation."""
from ._bootstrap import task_modules

_layout = task_modules()
AGENT_PROBLEM_FILENAME = _layout.AGENT_PROBLEM_FILENAME
AGENT_PROBLEM_SCHEMA_VERSION = _layout.AGENT_PROBLEM_SCHEMA_VERSION
GENERATED_AGENT_PROBLEM_FIELDS = _layout.GENERATED_AGENT_PROBLEM_FIELDS
GENERALIZED_AGENT_VISIBLE_FILES = _layout.GENERALIZED_AGENT_VISIBLE_FILES
LEGACY_ATREX_VISIBLE_FILES = _layout.LEGACY_ATREX_VISIBLE_FILES
is_sol_op = _layout.is_sol_op
find_atrex_bench_root = _layout.find_atrex_bench_root
validate_private_shapes = _layout.validate_private_shapes
validate_agent_problem = _layout.validate_agent_problem
validate_generated_agent_problem = _layout.validate_generated_agent_problem
has_agent_problem = _layout.has_agent_problem
should_use_generalized_problem = _layout.should_use_generalized_problem
agent_visible_operator_files = _layout.agent_visible_operator_files

__all__ = ['AGENT_PROBLEM_FILENAME', 'AGENT_PROBLEM_SCHEMA_VERSION', 'GENERATED_AGENT_PROBLEM_FIELDS', 'GENERALIZED_AGENT_VISIBLE_FILES', 'LEGACY_ATREX_VISIBLE_FILES', 'is_sol_op', 'find_atrex_bench_root', 'validate_private_shapes', 'validate_agent_problem', 'validate_generated_agent_problem', 'has_agent_problem', 'should_use_generalized_problem', 'agent_visible_operator_files']
