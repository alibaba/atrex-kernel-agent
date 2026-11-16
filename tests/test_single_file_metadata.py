import json

from aka.light.workspace.sources.single_file import SingleFile


def test_open_dsl_metadata_tracks_triton_candidate(tmp_path):
    (tmp_path / "kernel.py").write_text(
        "import torch\nimport triton\nimport triton.language as tl\n\ndef run(x):\n    return x\n"
    )

    source = SingleFile()
    source.seed_metadata(tmp_path, "", "vector_add", native_eval=True)

    metadata = json.loads((tmp_path / "solution.json").read_text())
    assert metadata["spec"]["languages"] == ["python", "triton"]
    assert metadata["spec"]["dependencies"] == ["torch", "triton"]
    assert metadata["dependency_roles"]["triton"] == (
        "Selected DSL and compiler for the candidate kernel."
    )
    assert source.language_hints(tmp_path) == ["python", "triton"]


def test_open_dsl_metadata_refreshes_after_language_selection(tmp_path):
    (tmp_path / "kernel.py").write_text("import torch\n\ndef run(x):\n    return x\n")
    source = SingleFile()
    source.seed_metadata(tmp_path, "", "vector_add", native_eval=True)
    assert json.loads((tmp_path / "solution.json").read_text())["spec"]["languages"] == [
        "python"
    ]

    (tmp_path / "kernel.py").write_text(
        "import torch\nfrom cutlass import cute\n\ndef run(x):\n    return x\n"
    )
    source.seed_metadata(tmp_path, "", "vector_add", native_eval=True)

    metadata = json.loads((tmp_path / "solution.json").read_text())
    assert metadata["spec"]["languages"] == ["python", "cutedsl"]
    assert metadata["spec"]["dependencies"] == ["torch", "nvidia-cutlass-dsl"]


def test_language_hints_reads_nested_manifest_spec(tmp_path):
    (tmp_path / "solution.json").write_text(
        json.dumps({"spec": {"languages": ["python", "flydsl"]}})
    )
    assert SingleFile().language_hints(tmp_path) == ["python", "flydsl"]


def test_pinned_cuda_removes_stale_supervisor_manifest(tmp_path):
    (tmp_path / "kernel.py").write_text("import torch\n")
    (tmp_path / "solution.json").write_text("{}\n")

    SingleFile().seed_metadata(tmp_path, "Cuda", "vector_add", native_eval=True)

    assert not (tmp_path / "solution.json").exists()
