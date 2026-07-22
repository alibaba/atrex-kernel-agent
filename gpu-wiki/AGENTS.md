# GPU Wiki Agent Entry

Read `README.md` first. For target-specific retrieval, use `python3 gpu-wiki/scripts/query.py` with explicit architecture/vendor/DSL filters before broad grep. Follow the self-containment contract: use relative in-wiki links, treat external checkouts as explicit environment variables, and run `python3 gpu-wiki/scripts/check-self-contained.py --root gpu-wiki` plus the query tests after path or index changes.
