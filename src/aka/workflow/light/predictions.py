"""Optional, host-frozen forecasts; never part of the GPU request or cache key."""
from datetime import UTC, datetime
from copy import deepcopy
from aka.experiment.measurement_store import private_write
from aka.experiment.measurement_store import read_json


def validate_predictions(value):
    if not isinstance(value, list):
        raise ValueError('predictions must be an array of {scope, prediction}')
    rows = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {'scope', 'prediction'}:
            raise ValueError('Each prediction requires exactly scope and prediction')
        if any(not isinstance(v, str) or not v.strip() for v in item.values()):
            raise ValueError('Prediction scope and prediction must be nonempty strings')
        rows.append({k: v.strip() for k, v in item.items()})
    return rows


def freeze_predictions(store, record_id, predictions):
    """Called while holding the task reservation, before execution starts.

    A recovered reservation must retain its original forecast, including an empty
    forecast: adding one after interrupted execution would not be preregistration.
    """
    path = store.root / 'records' / record_id / 'predictions.json'
    if path.exists():
        return read_json(path)
    frozen = {'recorded_at': datetime.now(UTC).isoformat(),
              'gateway_record_id': record_id, 'predictions': deepcopy(predictions)}
    private_write(path, frozen)
    return frozen
