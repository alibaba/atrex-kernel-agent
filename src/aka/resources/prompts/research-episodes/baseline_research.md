If essential framework/toolchain information is missing, make at most one query:

```bash
python3 tools/sandbox.py --kind wiki-query "<your description>" --brief
```

Include product `{{PLATFORM}}`, runtime architecture `{{ARCH}}`, operator, `{{FRAMEWORK}}`,
public shapes/dtypes and the missing fact. Inspect the first applicable record and stop research
once implementation is possible. Write your own implementation; do not delegate computation.
