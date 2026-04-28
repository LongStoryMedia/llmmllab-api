"""Models router - list available models."""

from typing import Optional

from fastapi import APIRouter, Query

from models import ModelTask
from utils.model_loader import ModelLoader

router = APIRouter()
model_loader = ModelLoader()


@router.get("/v1/models")
def list_models(task: Optional[str] = Query(default=None)):
    """List all available models, optionally filtered by task."""
    all_models = model_loader.get_available_models()
    models_list = []

    for model in all_models.values():
        if task and str(model.task) != task:
            continue
        models_list.append({
            "id": model.id,
            "name": model.name,
            "model": model.model,
            "task": str(model.task),
            "provider": str(model.provider),
            "digest": model.digest,
            "details": model.details.model_dump(mode="json"),
        })

    return models_list
