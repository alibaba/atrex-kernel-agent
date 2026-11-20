# Atrex Kernel Agent

AKA runs asynchronous kernel optimization tasks. An application supplies a source tree, a frozen correctness and timing contract, an execution target, and an objective. The research agent can profile and modify candidate kernels. The parent can reconnect to the task and export an accepted source revision.

Install the SDK with Python 3.10 or later:

```sh
python -m pip install .
aka task --help
aka execution --help
```

Use `aka.submit(..., execution={"backend": "local"})` for local experiments. For an existing Pod, select `remote` and supply its placement and transport settings. Research commands use the same target as evaluation. Use `aka.job(task_id)` to reconnect, `job.results()` to select a candidate and `job.export(candidate_id, destination)` to retrieve its source.

The [task interface](docs/task-execution.md) describes the contract, CLI, agent configuration and remote execution. [Native vector example](examples/task-contracts/native-vector/task.json).

To build the source distribution and wheel, run `python -m build`. Run the regression suite with `python -m unittest discover -s tests` after installing the SDK.
