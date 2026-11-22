"""Workflow-owned explicit input/output roles for auxiliary sessions."""

WORKSPACE_ROLE_ENV = "ATREX_AGENT_WORKSPACE_ROLE"
WORKSPACE_LAYOUTS = {
    "production-review": (("review_request.json", "candidate"), ("dependency_review.json",)),
    "problem-generation": (
        ("reference.py", "input.py", "shapes.json", "metadata.json"),
        ("agent_problem.json",),
    ),
    "research-claim-review": (
        ("submissions", "gateway-records", "references", "kernels", "evidence-manifest.json"),
        ("review.json",),
    ),
}
