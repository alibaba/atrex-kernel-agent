## GPU Wiki

At Episode start, query once with the exact operator identifier, target product, and runtime
architecture below (resolve an unknown architecture through the Runtime before querying):

```bash
python3 tools/sandbox.py --kind wiki-query "Target product {{PLATFORM}}, runtime architecture {{ARCH}}, DSL {{FRAMEWORK}}. Return the full product specification and techniques and pitfalls for operator {{OPERATOR}}." --brief
```

Additional targeted queries are allowed when new evidence raises a materially different question.
Treat knowledge as hypotheses to verify against Kernel/Gateway facts; do not copy Wiki metadata into
the Journal.
