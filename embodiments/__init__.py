"""One folder per robot. Each has `adapter.py` with `make(**options) -> core.embodiment.Embodiment`; `run.py --embodiment NAME` loads it.
Robot-specific programs, calibration, scripts and notes live beside the adapter in the same folder (see embodiments/g1/)."""
import importlib

KNOWN = ("mock", "g1", "agilex", "r1")


def load(name, **options):
    return importlib.import_module("embodiments.%s.adapter" % name).make(**options)
