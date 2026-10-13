"""Core registration for the independent observation contract."""
from aka.core.keys import ServiceKey

OBSERVATION_SOURCE = ServiceKey("observation_source", "ObservationSource",
                                module="aka.contracts.observation")
