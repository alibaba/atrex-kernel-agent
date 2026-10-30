At Episode start, make one conversion-specific Wiki query:

```bash
python3 tools/sandbox.py --kind wiki-query "Target product {{PLATFORM}}, runtime architecture {{ARCH}}, operator {{OPERATOR}}. Return the full product specification and only the matching Triton-to-Gluon conversion guidance." --brief
```

Resolve an unknown architecture through the Runtime before querying. Matching records are
`nvidia.blackwell.any.converter.blackwell` for `sm_100`/`sm_103`,
`nvidia.hopper.any.converter.hopper` for `sm_90`, `amd.cdna3.any.converter.cdna3` for `gfx94*`, and
`amd.cdna4.any.converter.cdna4` for `gfx95*`; never substitute a sibling architecture's record.
Treat guidance as hypotheses to verify. Additional targeted queries are allowed when new evidence
raises a materially different question.
