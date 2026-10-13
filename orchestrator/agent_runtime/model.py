"""Historical import routed through the external compatibility adapter."""
from orchestrator._bootstrap import task_modules
task_modules()
import importlib
import sys
sys.modules[__name__] = importlib.import_module('aka.legacy.compat.agent')
