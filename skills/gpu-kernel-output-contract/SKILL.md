---
name: gpu-kernel-output-contract
description: Final output contract for GPU kernel optimization tasks. Use this skill when a verified implementation must be packaged as generated_kernel.py for hidden evaluator import, especially when a stop hook blocks because the final candidate file is missing or non-compliant.
---

# GPU Kernel Output Contract

## SOL-ExecBench workspaces (definition.json + solution.json present)

If the workspace contains `definition.json` and `solution.json`, the output format is a **SOL solution**, NOT `generated_kernel.py`. Do not follow the `generated_kernel.py` / `class Model` contract below. Finalize deterministically with the backend script:

```bash
python reference/sol_finalize.py --workspace <workspace>
```

It packages the best `kernel.py` into a self-contained `submission.json` and re-validates it with the real `sol-execbench` evaluator over every workload (exit 0 ⇒ submittable). The rest of this document applies only to non-SOL (hidden-evaluator) workspaces.

---

This skill controls the final candidate packaging step for GPU kernel optimization work. It does not replace baseline implementation or profile-driven optimization. It is used when the workflow has a candidate implementation and must produce the exact file consumed by the hidden evaluator.

## When to Use

Use this skill when:

- A stop hook or goal-check hook asks you to convert the current candidate to the final output format.
- The optimization target is met but `generated_kernel.py` is missing.
- `generated_kernel.py` exists but contains tests, benchmark code, debug output, Markdown, prose, or other non-candidate content.
- The current implementation lives in development files such as `kernel.py`, `test_kernel.py`, reports, or profiles and must be packaged as the final candidate.

## Workflow and Evaluation Overview

- You may use the available tools and commands in the workspace to implement, test, benchmark, and iteratively refine the candidate before finalizing it.
- The final generated code will be evaluated by a hidden evaluator that is not provided directly.
- The hidden evaluator mainly checks whether the candidate compiles successfully, whether its outputs match the reference implementation, and how close its performance is to the hardware limit.
- These checks are applied sequentially: only candidates that compile successfully are checked for correctness, and only candidates that pass correctness are measured for performance.
- Optimize accordingly, but keep any temporary testing, benchmarking, profiling, or debugging workflow out of the final output file.

## Performance Expectations

- This is an optimization task, not only a functional translation task.
- The generated implementation will be evaluated on compile success, numerical correctness, and runtime performance.
- Among correct implementations, faster implementations are better.
- Do not degrade a validated optimized implementation while packaging it into `generated_kernel.py`.

## Output Contract

- Write the final implementation to `generated_kernel.py` in the project root directory (the parent of the `kernel_opt_*/` workspace), NOT inside the `kernel_opt_*/` subdirectory.
- `generated_kernel.py` is what will be evaluated; nothing printed to chat is read as the candidate.
- The file must contain exactly one self-contained Python module.
- `generated_kernel.py` must be independently executable/importable and must not depend on any other local `.py` file.
- The file must contain valid Python source only.
- Do not include Markdown fences, explanatory prose, tests, benchmarks, debug prints, or an `if __name__ == "__main__"` block.

## Implementation Requirements

- Include all necessary imports.
- Keep every required runtime helper, kernel, schedule, compiled program definition, and `Model` implementation inside `generated_kernel.py`.
- Do not import from or depend on any other local Python file.
- Define one or more GPU kernel implementations using any supported framework (Triton, Gluon, FlyDSL, CuteDSL, or C++ inline CUDA).
- Define `class Model(nn.Module)`.
- `Model.__init__` must accept the same initialization arguments as the reference `Model`.
- `Model.forward` must accept the same inputs as the reference `Model.forward`.
- `Model.forward` must return outputs with the same externally observable structure, shape, device, returned tensor dtype, and numerical behavior as the reference implementation.
- The main compute path must be implemented using GPU kernels from a supported framework (Triton, Gluon, FlyDSL, CuteDSL, or C++ inline CUDA) launched from `Model.forward`.
- You may use PyTorch only for setup or glue logic such as output allocation, reshape/view, indexing, metadata preparation, and launch orchestration.
- Internal computation precision, accumulation dtype, approximations, and intermediate layouts may differ from the PyTorch reference.
- You may use lower precision or mixed precision internally, including fp8, fp4, or int8 paths, if the returned outputs satisfy the evaluator's numerical tolerance for every evaluated shape.
- Do not call, instantiate, or wrap the original reference `Model`.
- Do not use `torch.compile`, `torch.jit`, custom C++ extensions, or external files.

## Correctness Requirements

- Preserve the reference's externally observable semantics exactly.
- Treat internal casts and accumulation choices in the PyTorch reference as defining the numerical target, not as mandatory implementation choices.
- The generated implementation may use different internal precision or accumulation strategies if final outputs pass correctness.
- Preserve all important masking, routing, indexing, and output-layout behavior implied by the reference code.
- If multiple kernels, schedules, or programs are needed, keep them in the same file and call them from `Model.forward`.

## Evaluation Contract

- The generated file will be imported directly by the evaluator.
- The evaluator reads init kwargs and input kwargs from `shapes.json`, one entry per shape keyed by id, sitting next to `reference.py`.
- Each shape entry has `init_kwargs`, passed to `Model(**init_kwargs)`, and `input_kwargs`, passed to `_make_inputs(**input_kwargs)` defined in `input.py`.
- `_make_inputs(**input_kwargs)` returns `dict[str, Tensor]`.
- The evaluator then calls `Model(**init_kwargs)(**call_inputs)`, so `Model.forward` parameter names must match the keys returned by `_make_inputs`.
- The generated candidate file only needs to provide `class Model`.
- Do not redefine `_make_inputs`.
- Do not read `shapes.json`.
- If the reference returns a single tensor, the candidate must also return a single tensor with the same dtype and shape.
- If the reference returns multiple tensors as a `dict[str, Tensor]`, the candidate must return a dict with the same keys.

## Forbidden Content

The final `generated_kernel.py` must not contain:

- Any operator-specific optimization hint not inferable from the reference itself.
- Explanatory prose.
- Markdown fences or Markdown formatting.
- Benchmark code.
- Test code.
- Debug prints.
- Placeholder unfinished content.
- An `if __name__ == "__main__"` block.
- A fallback that uses the original reference `Model` as the primary execution path.
- Calls to `torch.compile` or `torch.jit`.
- Custom C++ extension build or load logic.
- Reads from external files such as `shapes.json`, `input.py`, profiling artifacts, reports, or temporary files.
- Imports from or runtime dependencies on other local `.py` files.

## Final Packaging Procedure

1. Identify the current validated optimized implementation and its reference `Model` signature.
2. Copy only the runtime implementation needed by the candidate into `generated_kernel.py`.
3. Keep GPU kernel definitions (Triton, Gluon, FlyDSL, CuteDSL, or C++ inline CUDA), schedules, compiled program definitions, helper functions, and `class Model(nn.Module)` in the same file.
4. Preserve `Model.__init__` and `Model.forward` signatures required by the evaluator.
5. Remove any tests, benchmarks, prints, profiling code, report generation, command-line parsing, and `__main__` entry.
6. Run a syntax check outside `generated_kernel.py` before stopping.
7. Run import, correctness, and performance checks outside the final file whenever the required runtime dependencies and local reference inputs are present.
8. If any required dependency, input, or evaluator-adjacent file is unavailable, record that limitation outside `generated_kernel.py` and still keep the candidate file clean.
9. Leave the final candidate as `generated_kernel.py` only.

## Final Checklist

Before stopping, verify:

- `generated_kernel.py` exists in the project root directory (parent of `kernel_opt_*/`), not inside the workspace subdirectory.
- It is valid Python source.
- It imports everything it needs.
- It is independently executable/importable and does not depend on any other local `.py` file.
- It defines `class Model(nn.Module)`.
- `Model.__init__` matches the reference initialization arguments.
- `Model.forward` matches the evaluator input keys.
- The main compute path launches GPU kernels from a supported framework (Triton, Gluon, FlyDSL, CuteDSL, or C++ inline CUDA) from `Model.forward`.
- The returned structure, keys, shapes, device, dtype, and numerical behavior match the reference.
- The file does not read `shapes.json` or redefine `_make_inputs`.
- The file does not call, instantiate, or wrap the original reference `Model`.
- The file contains no Markdown, prose, tests, benchmarks, debug prints, unfinished content, or `__main__` block.
