"""
Standalone data models for llmmllab-runner.

These types are duplicated from the main app's models/ package so the runner
service has zero runtime dependencies on the main application.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat, constr


# --- Enums ---


class ModelProvider(str, Enum):
    LLAMA_CPP = "llama_cpp"
    HF = "hf"
    HUGGING_FACE = "hugging_face"
    OPENAI = "openai"
    STABLE_DIFFUSION_CPP = "stable_diffusion_cpp"
    ANTHROPIC = "anthropic"
    OTHER = "other"


class ModelTask(str, Enum):
    TEXTTOTEXT = "TextToText"
    TEXTTOIMAGE = "TextToImage"
    IMAGETOTEXT = "ImageToText"
    IMAGETOIMAGE = "ImageToImage"
    TEXTTOAUDIO = "TextToAudio"
    AUDIOTOTEXT = "AudioToText"
    TEXTTOVIDEO = "TextToVideo"
    VIDEOTOTEXT = "VideoToText"
    TEXTTOSPEECH = "TextToSpeech"
    SPEECHTOTEXT = "SpeechToText"
    TEXTTOEMBEDDINGS = "TextToEmbeddings"
    VISIONTEXTTOTEXT = "VisionTextToText"
    IMAGETEXTTOIMAGE = "ImageTextToImage"
    TEXTTORANKING = "TextToRanking"
    IMAGETO3D = "ImageTo3D"


class PipelinePriority(Enum):
    LOW = 1
    MEDIUM = 5
    NORMAL = 10
    HIGH = 11
    CRITICAL = 20


# --- Supporting models ---


class LoraWeight(BaseModel):
    id: Annotated[str, Field(..., description="Unique identifier for the LoRA weight")]
    name: Annotated[str, Field(..., description="Name of the LoRA weight")]
    parent_model: Annotated[str, Field(..., description="Identifier of the parent model this LoRA weight is associated with")]
    weight_name: Annotated[Optional[str], Field(default=None, description="Name of the weight file")] = None
    adapter_name: Annotated[Optional[str], Field(default=None, description="Name of the adapter")] = None

    model_config = ConfigDict(extra="ignore")


class ModelDetails(BaseModel):
    parent_model: Annotated[Optional[str], Field(default=None, description="Identifier of the parent model if this is a derivative")] = None
    format: Annotated[str, Field(..., description="Format of the model file (e.g., gguf)")]
    gguf_file: Annotated[Optional[str], Field(default=None, description="Path to the GGUF file if applicable")] = None
    clip_model_path: Annotated[Optional[str], Field(default=None, description="Path to the CLIP model file for multimodal models")] = None
    family: Annotated[str, Field(..., description="Primary model family this belongs to")]
    families: Annotated[List[str], Field(..., description="All model families this belongs to")]
    parameter_size: Annotated[str, Field(..., description="Size of model parameters (e.g., '7.2B')")]
    quantization_level: Annotated[Optional[str], Field(default=None, description="Level of quantization applied to the model")] = None
    dtype: Annotated[Optional[str], Field(default=None, description="Data type of the model")] = None
    precision: Annotated[Optional[Literal["fp32", "fp16", "bf16", "int8", "int4", "int2", "int1"]], Field(default=None, description="Precision of the model")] = None
    specialization: Annotated[Optional[Literal["LoRA", "Embedding", "TextToImage", "ImageToImage", "Audio", "Text"]], Field(default=None, description="Specialization of the model")] = None
    description: Annotated[Optional[str], Field(default=None, description="Description of the model")] = None
    weight: Annotated[Optional[float], Field(default=None, description="Weight of the model (applies to LoRA models)")] = None
    size: Annotated[int, Field(..., description="Size of the model in bytes")]
    original_ctx: Annotated[int, Field(..., description="Original context window size of the model")]
    n_layers: Annotated[Optional[int], Field(default=None, description="Number of layers in the model")] = None
    hidden_size: Annotated[Optional[int], Field(default=None, description="Hidden size of the model")] = None
    n_heads: Annotated[Optional[int], Field(default=None, description="Number of attention heads in the model")] = None
    n_kv_heads: Annotated[Optional[int], Field(default=None, description="Number of key-value heads in the model")] = None
    clip_model_size: Annotated[Optional[int], Field(default=None, description="Size of the CLIP model in bytes, if applicable")] = None

    model_config = ConfigDict(extra="ignore")


class ModelParameters(BaseModel):
    num_ctx: Annotated[Optional[int], Field(default=None, description="Size of the context window")] = None
    repeat_last_n: Annotated[Optional[int], Field(default=None, description="Number of tokens to consider for repetition penalties")] = None
    repeat_penalty: Annotated[Optional[float], Field(default=None, description="Penalty for repetitions")] = None
    temperature: Annotated[Optional[float], Field(default=None, description="Sampling temperature")] = None
    seed: Annotated[Optional[int], Field(default=None, description="Random seed for reproducibility")] = None
    stop: Annotated[Optional[List[str]], Field(default=None, description="Sequences where the model should stop generating")] = None
    num_predict: Annotated[Optional[int], Field(default=None, description="Maximum number of tokens to predict")] = None
    top_k: Annotated[Optional[int], Field(default=None, description="Limits next token selection to top K options")] = None
    top_p: Annotated[Optional[float], Field(default=None, description="Nucleus sampling probability mass")] = None
    min_p: Annotated[Optional[float], Field(default=None, description="Minimum probability threshold for token selection")] = None
    think: Annotated[Optional[bool], Field(default=None, description='Whether to enable "thinking" mode for the model')] = None
    max_tokens: Annotated[Optional[int], Field(default=None, description="Maximum number of tokens to generate")] = None
    n_parts: Annotated[Optional[int], Field(default=None, description="Number of parts to split the model into. -1 means auto.")] = None
    batch_size: Annotated[Optional[int], Field(default=None, description="Batch size for processing inputs")] = None
    micro_batch_size: Annotated[Optional[int], Field(default=None, description="Micro batch size for processing inputs")] = None
    n_gpu_layers: Annotated[Optional[int], Field(default=None, description="Number of model layers to keep on GPU", ge=-1)] = None
    main_gpu: Annotated[Optional[int], Field(default=-1, description="Main GPU device index (-1 for auto-selection)", ge=-1)] = -1
    tensor_split: Annotated[Optional[str], Field(default=None, description="Comma-separated fractions of model to put on each GPU")] = None
    split_mode: Annotated[Optional[Literal["none", "layer", "row"]], Field(default="layer", description="How to split model across devices")] = "layer"
    n_cpu_moe: Annotated[Optional[int], Field(default=0, description="Number of MoE layers to keep on CPU", ge=0)] = 0
    kv_on_cpu: Annotated[Optional[bool], Field(default=False, description="Store key-value cache on CPU to save GPU memory")] = False
    reasoning_effort: Annotated[Optional[Literal["low", "medium", "high"]], Field(default="medium", description="Reasoning effort level")] = "medium"
    flash_attention: Annotated[Optional[bool], Field(default=True, description="Enable flash attention optimization")] = True

    model_config = ConfigDict(extra="ignore")


class ModelProfileImageSettings(BaseModel):
    height: Annotated[Optional[int], Field(default=None, description="Height of the image in pixels")] = None
    width: Annotated[Optional[int], Field(default=None, description="Width of the image in pixels")] = None
    inference_steps: Annotated[Optional[int], Field(default=None, description="Number of inference steps")] = None
    guidance_scale: Annotated[Optional[float], Field(default=None, description="Guidance scale for image generation")] = None
    low_memory_mode: Annotated[Optional[bool], Field(default=None, description="Whether to use low memory mode")] = None
    negative_prompt: Annotated[Optional[str], Field(default=None, description="Negative prompt for image generation")] = None
    lora_model: Annotated[Optional[str], Field(default=None, description="Name of the LoRA model to use for image generation")] = None

    model_config = ConfigDict(extra="ignore")


# --- Main model ---


class Model(BaseModel):
    id: Annotated[Optional[str], Field(default=None, description="Unique identifier for the model")] = None
    name: Annotated[str, Field(..., description="Display name of the model")]
    model: Annotated[str, Field(..., description="Identifier used to reference the model")]
    task: Annotated[ModelTask, Field(..., description="Type of task the model is designed for")]
    modified_at: Annotated[str, Field(..., description="Timestamp of when the model was last modified")]
    digest: Annotated[str, Field(..., description="Hash digest identifying the model version")]
    details: Annotated[ModelDetails, Field(..., description="Additional information about the model")]
    pipeline: Annotated[Optional[str], Field(default=None, description="Pipeline type used for the model")] = None
    lora_weights: Annotated[Optional[List[LoraWeight]], Field(default=None, description="List of LoRA weights associated with the model")] = None
    provider: Annotated[ModelProvider, Field(..., description="Provider or runtime of the model")]
    system_prompt: Annotated[Optional[str], Field(default=None, description="System prompt to use when running this model")] = None
    parameters: Annotated[Optional[ModelParameters], Field(default=None, description="Default inference parameters for this model")] = None
    image_settings: Annotated[Optional[ModelProfileImageSettings], Field(default=None, description="Image generation settings (for image models)")] = None
    draft_model: Annotated[Optional[str], Field(default=None, description="Optional draft model for speculative decoding")] = None

    model_config = ConfigDict(extra="ignore")


# --- User config (simplified - runner only needs model_id) ---


class UserConfig(BaseModel):
    """Simplified user config for runner server argument building."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())
