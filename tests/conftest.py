"""Shared test setup: pipeline/ on the import path and the 'gpu' marker."""
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(CODE_DIR, "pipeline")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: loads real checkpoints or features (slow)")
