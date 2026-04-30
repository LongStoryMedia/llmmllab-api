from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Union
from middleware.auth import get_user_id
from services.runner_client import runner_client
from models.openai import DeleteModelResponse, ListModelsResponse, Model as OpenAIModel
from models.anthropic import (
    ModelListResponse as AnthropicModelListResponse,
    Model as AnthropicModel,
)
from models.model import Model
from models.model_task import ModelTask
from models.model_details import ModelDetails
from models.model_provider import ModelProvider
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="common_models_router")
router = APIRouter(prefix="/models", tags=["Models"])


# Union types for common endpoints
OpenAIModelListResponse = ListModelsResponse

OpenAIModelType = OpenAIModel
AnthropicModelType = AnthropicModel


def to_openai_model(model_id: str) -> OpenAIModel:
    """Create an OpenAI API model response from a model id."""
    return OpenAIModel(
        id=model_id,
        object="model",
        created=int(datetime.now().timestamp()),
        owned_by="llmmllab",
    )


def to_anthropic_model(model_id: str, display_name: str) -> AnthropicModel:
    """Create an Anthropic API model response from a model id."""
    return AnthropicModel(
        id=model_id,
        type="model",
        display_name=display_name,
        created_at=datetime.now(),
    )


def from_openai_model(
    openai_model: OpenAIModel,
    task: ModelTask = ModelTask.TEXTTOTEXT,
) -> Model:
    """Convert OpenAI API model format to internal Model representation."""
    return Model(
        id=openai_model.id,
        name=openai_model.id,
        model=openai_model.id,
        task=task,
        modified_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        digest=openai_model.id,
        details=ModelDetails(
            format="",
            family="",
            families=[],
            parameter_size="",
            size=0,
            original_ctx=0,
        ),
        provider=ModelProvider.LLAMA_CPP,  # TODO: what are the implications here?
    )


@router.get("/")
async def listModels(request: Request) -> ListModelsResponse:
    """Operation ID: listModels"""
    # We're not currently using the user_id for filtering, but we may in the future
    _ = get_user_id(request)

    try:
        models = await runner_client.list_models()

        logger.info(f"Successfully loaded {len(models)} models for API")
        model_ids = [m.id for m in models if m.id]
        return ListModelsResponse(
            data=[to_openai_model(mid) for mid in model_ids],
            object="list",
        )

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error loading models: {str(e)}"
        ) from e


@router.get("/{model_id}")
async def getModel(
    model_id: str,
    request: Request,
) -> Union[OpenAIModelType, AnthropicModelType]:
    """Operation ID: getModel (OpenAI) / getModel (Anthropic)"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{model_id}")
async def deleteModel(
    model_id: str,
    request: Request,
) -> DeleteModelResponse:
    """Operation ID: deleteModel (OpenAI)"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")
