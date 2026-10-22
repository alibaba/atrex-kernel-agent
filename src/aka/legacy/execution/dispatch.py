"""Execute an ordered group of independent physical requests and join every worker."""
from concurrent.futures import ThreadPoolExecutor


def run_batches(items, operation, *, max_workers):
    if not items:
        return []
    if len(items) == 1:
        return [operation(items[0])]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(operation, items))
