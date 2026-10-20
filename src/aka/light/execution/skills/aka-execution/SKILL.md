---
name: aka-execution
description: Use the selected reliable GPU executor from the legacy Workflow.
---

For prepared host commands, use the selected `aka execution --profile reliable` entry point and its `--help` request format. The selected controller executor retains request checkpoints and known remote job IDs outside the Agent workspace. Retries apply only to confirmed admission/transport failures. An unknown submission outcome must be reconciled by the operator, not blindly resubmitted. An Eval execution timeout is a candidate failure, not an environment outage. Do not switch a failed Eval to Dev.

This selection does not replace a legacy Agent's direct `tools/sandbox.py`, or install the light HTTP ToolHost or Journal. The light Workflow separately provides its authenticated HTTP client and the `gpu-measurement` and `runtime-records` Skills. Do not assume that loading this Skill changes every tool's transport.
