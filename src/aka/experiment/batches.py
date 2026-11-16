"""Original typed-gateway batch ordering and parallel execution."""
from concurrent.futures import ThreadPoolExecutor


def run_batches(items, operation, *, max_workers):
    """Keep the typed route's ordered map and wait-for-all failure semantics.

    Unlike formal ABBA, this route does not install an outage cancellation Event
    or copy contextvars. A single batch still runs in the caller's thread.
    """
    if len(items) <= 1:
        return [operation(items[0])]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(operation, items))
