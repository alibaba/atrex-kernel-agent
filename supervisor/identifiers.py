"""Journal identifiers share the Measurement Store's existing Kernel/Record IDs."""

import re

from supervisor.measurement_records import KERNEL_ID, RECORD_ID

DIRECTION_ID_RE = re.compile(r"direction_[0-9a-f]{32}")
EXPERIMENT_ID_RE = re.compile(r"experiment_[0-9a-f]{32}")
GATEWAY_RECORD_ID_RE = RECORD_ID
KERNEL_RECORD_ID_RE = KERNEL_ID
