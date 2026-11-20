import contextlib
import io
import unittest
from unittest.mock import patch

from aka.cli import main


class TaskCliTests(unittest.TestCase):
    def test_help_lists_only_task_commands(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["--help"]), 0)
        self.assertEqual(out.getvalue().strip(), "usage: aka {task,execution} [ARGS ...]")

    def test_retired_commands_cannot_start(self):
        for name in ("run", "workflow", "optimize"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([name]), 2)

    def test_task_and_execution_receive_original_arguments(self):
        for name, module in (("task", "aka.light.tasks.cli"), ("execution", "aka.light.execution.command_cli")):
            with self.subTest(name=name), patch(module + ".main", return_value=17) as target:
                self.assertEqual(main([name, "--help"]), 17)
                target.assert_called_once_with(["--help"])
