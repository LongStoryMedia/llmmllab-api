from typing import Dict

from models.model import Model


def load_test_models() -> Dict[str, Model]:
    """
    Load models for testing purposes.

    Returns:
        Dict[str, Model]: An empty dictionary.

    Models are now managed by the llmmllab-runner service.
    Test fixtures in conftest.py provide inline Model instances.
    """
    return {}
