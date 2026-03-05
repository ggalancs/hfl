# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Detection and validation of model formats."""

from enum import Enum
from pathlib import Path


class ModelFormat(Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Type of model based on its functionality."""

    # Supported types
    LLM = "llm"  # Text generation models
    TTS = "tts"  # Text-to-speech models

    # Unsupported types (detected but not runnable)
    STT = "stt"  # Speech-to-text / Automatic Speech Recognition
    IMAGE_GEN = "image-generation"  # Text-to-image (Stable Diffusion, etc.)
    IMAGE_CLASS = "image-classification"  # Image classification
    OBJECT_DETECT = "object-detection"  # Object detection
    IMAGE_SEG = "image-segmentation"  # Image segmentation
    VIDEO = "video"  # Video models
    EMBEDDING = "embedding"  # Text/image embeddings
    FILL_MASK = "fill-mask"  # Masked language models (BERT, etc.)
    TOKEN_CLASS = "token-classification"  # NER, POS tagging
    QA = "question-answering"  # Extractive QA
    SUMMARIZATION = "summarization"  # Text summarization
    TRANSLATION = "translation"  # Machine translation
    FEATURE_EXTRACT = "feature-extraction"  # Feature extraction
    SENTENCE_SIM = "sentence-similarity"  # Sentence similarity
    ZERO_SHOT = "zero-shot-classification"  # Zero-shot classification
    TABLE_QA = "table-question-answering"  # Table QA
    VISUAL_QA = "visual-qa"  # Visual question answering
    DOCUMENT_QA = "document-qa"  # Document question answering
    IMAGE_TEXT = "image-to-text"  # Image captioning
    DEPTH = "depth-estimation"  # Depth estimation
    MULTIMODAL = "multimodal"  # General multimodal models
    REINFORCEMENT = "reinforcement-learning"  # RL models
    ROBOTICS = "robotics"  # Robotics models
    AUDIO_CLASS = "audio-classification"  # Audio classification

    UNKNOWN = "unknown"


# Types that HFL can run
SUPPORTED_MODEL_TYPES = {ModelType.LLM, ModelType.TTS}


def is_model_type_supported(model_type: ModelType) -> bool:
    """Check if a model type is supported by HFL."""
    return model_type in SUPPORTED_MODEL_TYPES


def model_type_from_pipeline_tag(pipeline_tag: str | None) -> ModelType | None:
    """Convert a HuggingFace pipeline_tag to ModelType.

    Args:
        pipeline_tag: The pipeline_tag from HuggingFace model info

    Returns:
        ModelType if recognized, None otherwise
    """
    if not pipeline_tag:
        return None
    return PIPELINE_TAG_TO_TYPE.get(pipeline_tag)


def get_model_type_display_name(model_type: ModelType) -> str:
    """Get human-readable name for a model type."""
    display_names = {
        ModelType.LLM: "LLM (Text Generation)",
        ModelType.TTS: "TTS (Text-to-Speech)",
        ModelType.STT: "STT (Speech-to-Text)",
        ModelType.IMAGE_GEN: "Image Generation",
        ModelType.IMAGE_CLASS: "Image Classification",
        ModelType.OBJECT_DETECT: "Object Detection",
        ModelType.IMAGE_SEG: "Image Segmentation",
        ModelType.VIDEO: "Video",
        ModelType.EMBEDDING: "Embeddings",
        ModelType.FILL_MASK: "Fill-Mask (MLM)",
        ModelType.TOKEN_CLASS: "Token Classification",
        ModelType.QA: "Question Answering",
        ModelType.SUMMARIZATION: "Summarization",
        ModelType.TRANSLATION: "Translation",
        ModelType.FEATURE_EXTRACT: "Feature Extraction",
        ModelType.SENTENCE_SIM: "Sentence Similarity",
        ModelType.ZERO_SHOT: "Zero-Shot Classification",
        ModelType.TABLE_QA: "Table QA",
        ModelType.VISUAL_QA: "Visual QA",
        ModelType.DOCUMENT_QA: "Document QA",
        ModelType.IMAGE_TEXT: "Image-to-Text",
        ModelType.DEPTH: "Depth Estimation",
        ModelType.MULTIMODAL: "Multimodal",
        ModelType.REINFORCEMENT: "Reinforcement Learning",
        ModelType.ROBOTICS: "Robotics",
        ModelType.AUDIO_CLASS: "Audio Classification",
        ModelType.UNKNOWN: "Unknown",
    }
    return display_names.get(model_type, model_type.value)


def detect_format(model_path: Path) -> ModelFormat:
    """Detects the format of a model given its directory or file."""
    if model_path.is_file():
        if model_path.suffix == ".gguf":
            return ModelFormat.GGUF
        elif model_path.suffix == ".safetensors":
            return ModelFormat.SAFETENSORS
        elif model_path.suffix in (".pt", ".pth", ".bin"):
            return ModelFormat.PYTORCH

    if model_path.is_dir():
        files = list(model_path.rglob("*"))
        extensions = {f.suffix for f in files}

        if ".gguf" in extensions:
            return ModelFormat.GGUF
        elif ".safetensors" in extensions:
            return ModelFormat.SAFETENSORS
        elif ".bin" in extensions or ".pt" in extensions:
            return ModelFormat.PYTORCH

    return ModelFormat.UNKNOWN


def find_model_file(model_path: Path, fmt: ModelFormat) -> Path | None:
    """Finds the main model file."""
    if model_path.is_file():
        return model_path

    if fmt == ModelFormat.GGUF:
        gguf_files = list(model_path.rglob("*.gguf"))
        return gguf_files[0] if gguf_files else None

    return model_path  # For safetensors/pytorch, return the directory


# =============================================================================
# Architecture and Pipeline Tag Mappings
# =============================================================================

# LLM architectures (text generation)
LLM_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "GPT2LMHeadModel",
    "GPTNeoForCausalLM",
    "GPTNeoXForCausalLM",
    "GPTJForCausalLM",
    "BloomForCausalLM",
    "FalconForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "StableLmForCausalLM",
    "OPTForCausalLM",
    "MambaForCausalLM",
    "RecurrentGemmaForCausalLM",
    "CohereForCausalLM",
    "CommandRForCausalLM",
    "DeepseekV2ForCausalLM",
    "InternLMForCausalLM",
    "InternLM2ForCausalLM",
    "BaichuanForCausalLM",
    "YiForCausalLM",
    "StarCoder2ForCausalLM",
    "CodeLlamaForCausalLM",
    "CodeGenForCausalLM",
    "ArcticForCausalLM",
    "JambaForCausalLM",
    "OlmoForCausalLM",
    "Olmo2ForCausalLM",
}

# TTS architectures (text-to-speech)
TTS_ARCHITECTURES = {
    "BarkModel",
    "BarkSemanticModel",
    "BarkCoarseModel",
    "BarkFineModel",
    "SpeechT5ForTextToSpeech",
    "SpeechT5Model",
    "VitsModel",
    "FastSpeech2ConformerModel",
    "SeamlessM4TModel",
    "SeamlessM4TForTextToSpeech",
    "MmsModel",
    "MmsTtsModel",
    "ParlerTTSForConditionalGeneration",
}

# STT architectures (speech-to-text / ASR)
STT_ARCHITECTURES = {
    "WhisperForConditionalGeneration",
    "Wav2Vec2ForCTC",
    "Wav2Vec2ForSequenceClassification",
    "HubertForCTC",
    "Speech2TextForConditionalGeneration",
    "SpeechT5ForSpeechToText",
    "SeamlessM4TForSpeechToText",
    "WhisperModel",
    "Wav2Vec2Model",
    "WavLMForCTC",
    "Data2VecAudioForCTC",
    "MCTCTForCTC",
    "UniSpeechForCTC",
    "UniSpeechSatForCTC",
}

# Image generation architectures
IMAGE_GEN_ARCHITECTURES = {
    "StableDiffusionPipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusion3Pipeline",
    "FluxPipeline",
    "KandinskyPipeline",
    "UNet2DConditionModel",
    "AutoencoderKL",
    "DiTPipeline",
    "PixArtAlphaPipeline",
    "PixArtSigmaPipeline",
}

# Image classification architectures
IMAGE_CLASS_ARCHITECTURES = {
    "ViTForImageClassification",
    "ResNetForImageClassification",
    "ConvNextForImageClassification",
    "ConvNextV2ForImageClassification",
    "SwinForImageClassification",
    "BeitForImageClassification",
    "DeiTForImageClassification",
    "EfficientNetForImageClassification",
    "MobileNetV2ForImageClassification",
    "MobileViTForImageClassification",
    "RegNetForImageClassification",
    "PoolFormerForImageClassification",
    "DPTForDepthEstimation",
}

# Object detection architectures
OBJECT_DETECT_ARCHITECTURES = {
    "DetrForObjectDetection",
    "YolosForObjectDetection",
    "ConditionalDetrForObjectDetection",
    "DeformableDetrForObjectDetection",
    "DetaForObjectDetection",
    "RTDetrForObjectDetection",
    "GroundingDinoForObjectDetection",
    "OwlViTForObjectDetection",
    "Owlv2ForObjectDetection",
}

# Image segmentation architectures
IMAGE_SEG_ARCHITECTURES = {
    "MaskFormerForInstanceSegmentation",
    "Mask2FormerForUniversalSegmentation",
    "SegformerForSemanticSegmentation",
    "UperNetForSemanticSegmentation",
    "BeitForSemanticSegmentation",
    "DetrForSegmentation",
    "OneFormerForUniversalSegmentation",
    "SamModel",
    "CLIPSegForImageSegmentation",
}

# Embedding architectures
EMBEDDING_ARCHITECTURES = {
    "BertModel",
    "RobertaModel",
    "DistilBertModel",
    "AlbertModel",
    "XLMRobertaModel",
    "MPNetModel",
    "DebertaModel",
    "DebertaV2Model",
    "SentenceTransformer",
    "E5Model",
    "GTEModel",
    "BGEModel",
    "CLIPModel",
    "CLIPTextModel",
    "SiglipModel",
}

# Fill-mask (MLM) architectures
FILL_MASK_ARCHITECTURES = {
    "BertForMaskedLM",
    "RobertaForMaskedLM",
    "DistilBertForMaskedLM",
    "AlbertForMaskedLM",
    "XLMRobertaForMaskedLM",
    "DebertaForMaskedLM",
    "DebertaV2ForMaskedLM",
    "CamembertForMaskedLM",
    "ElectraForMaskedLM",
}

# Token classification (NER, POS) architectures
TOKEN_CLASS_ARCHITECTURES = {
    "BertForTokenClassification",
    "RobertaForTokenClassification",
    "DistilBertForTokenClassification",
    "AlbertForTokenClassification",
    "XLMRobertaForTokenClassification",
    "DebertaForTokenClassification",
    "DebertaV2ForTokenClassification",
    "CamembertForTokenClassification",
    "LayoutLMForTokenClassification",
    "LayoutLMv2ForTokenClassification",
    "LayoutLMv3ForTokenClassification",
}

# Question answering architectures
QA_ARCHITECTURES = {
    "BertForQuestionAnswering",
    "RobertaForQuestionAnswering",
    "DistilBertForQuestionAnswering",
    "AlbertForQuestionAnswering",
    "XLMRobertaForQuestionAnswering",
    "DebertaForQuestionAnswering",
    "DebertaV2ForQuestionAnswering",
    "LongformerForQuestionAnswering",
    "BigBirdForQuestionAnswering",
}

# Summarization/Translation (Seq2Seq) architectures
SEQ2SEQ_ARCHITECTURES = {
    "T5ForConditionalGeneration",
    "MT5ForConditionalGeneration",
    "BartForConditionalGeneration",
    "MBartForConditionalGeneration",
    "PegasusForConditionalGeneration",
    "MarianMTModel",
    "M2M100ForConditionalGeneration",
    "NllbMoeForConditionalGeneration",
    "BlenderbotForConditionalGeneration",
    "LEDForConditionalGeneration",
    "LongT5ForConditionalGeneration",
}

# Visual QA / Image-Text architectures
VISUAL_QA_ARCHITECTURES = {
    "BlipForQuestionAnswering",
    "Blip2ForConditionalGeneration",
    "ViltForQuestionAnswering",
    "VisualBertForQuestionAnswering",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "InstructBlipForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "CogVLMForCausalLM",
    "Phi3VForCausalLM",
    "InternVLChatModel",
}

# Image-to-text (captioning) architectures
IMAGE_TEXT_ARCHITECTURES = {
    "BlipForConditionalGeneration",
    "GitForCausalLM",
    "VisionEncoderDecoderModel",
    "TrOCRForCausalLM",
    "Pix2StructForConditionalGeneration",
    "DonutProcessor",
    "Florence2ForConditionalGeneration",
}

# Depth estimation architectures
DEPTH_ARCHITECTURES = {
    "DPTForDepthEstimation",
    "GLPNForDepthEstimation",
    "ZoeDepthForDepthEstimation",
    "DepthAnythingForDepthEstimation",
}

# Audio classification architectures
AUDIO_CLASS_ARCHITECTURES = {
    "Wav2Vec2ForSequenceClassification",
    "HubertForSequenceClassification",
    "WavLMForSequenceClassification",
    "Data2VecAudioForSequenceClassification",
    "ASTForAudioClassification",
    "WhisperForAudioClassification",
}

# Video architectures
VIDEO_ARCHITECTURES = {
    "VideoMAEForVideoClassification",
    "TimesformerForVideoClassification",
    "VivitForVideoClassification",
    "CogVideoXPipeline",
    "VideoLlavaForConditionalGeneration",
}

# Pipeline tag to ModelType mapping
PIPELINE_TAG_TO_TYPE = {
    # LLM
    "text-generation": ModelType.LLM,
    "conversational": ModelType.LLM,
    # TTS
    "text-to-speech": ModelType.TTS,
    "text-to-audio": ModelType.TTS,
    # STT
    "automatic-speech-recognition": ModelType.STT,
    "audio-to-audio": ModelType.STT,
    # Image generation
    "text-to-image": ModelType.IMAGE_GEN,
    "image-to-image": ModelType.IMAGE_GEN,
    "unconditional-image-generation": ModelType.IMAGE_GEN,
    # Image understanding
    "image-classification": ModelType.IMAGE_CLASS,
    "object-detection": ModelType.OBJECT_DETECT,
    "image-segmentation": ModelType.IMAGE_SEG,
    "depth-estimation": ModelType.DEPTH,
    "image-to-text": ModelType.IMAGE_TEXT,
    # Audio
    "audio-classification": ModelType.AUDIO_CLASS,
    # Video
    "video-classification": ModelType.VIDEO,
    "text-to-video": ModelType.VIDEO,
    # NLP tasks
    "fill-mask": ModelType.FILL_MASK,
    "token-classification": ModelType.TOKEN_CLASS,
    "question-answering": ModelType.QA,
    "summarization": ModelType.SUMMARIZATION,
    "translation": ModelType.TRANSLATION,
    "text2text-generation": ModelType.SUMMARIZATION,
    # Embeddings
    "feature-extraction": ModelType.EMBEDDING,
    "sentence-similarity": ModelType.SENTENCE_SIM,
    # Zero-shot
    "zero-shot-classification": ModelType.ZERO_SHOT,
    "zero-shot-image-classification": ModelType.ZERO_SHOT,
    "zero-shot-object-detection": ModelType.ZERO_SHOT,
    # QA variants
    "table-question-answering": ModelType.TABLE_QA,
    "visual-question-answering": ModelType.VISUAL_QA,
    "document-question-answering": ModelType.DOCUMENT_QA,
    # Other
    "reinforcement-learning": ModelType.REINFORCEMENT,
    "robotics": ModelType.ROBOTICS,
}

# Model type field to ModelType mapping
MODEL_TYPE_FIELD_TO_TYPE = {
    # LLM
    "llama": ModelType.LLM,
    "mistral": ModelType.LLM,
    "mixtral": ModelType.LLM,
    "qwen2": ModelType.LLM,
    "qwen2_moe": ModelType.LLM,
    "gpt2": ModelType.LLM,
    "gpt_neo": ModelType.LLM,
    "gpt_neox": ModelType.LLM,
    "gptj": ModelType.LLM,
    "bloom": ModelType.LLM,
    "falcon": ModelType.LLM,
    "phi": ModelType.LLM,
    "phi3": ModelType.LLM,
    "gemma": ModelType.LLM,
    "gemma2": ModelType.LLM,
    "stablelm": ModelType.LLM,
    "opt": ModelType.LLM,
    "mamba": ModelType.LLM,
    "cohere": ModelType.LLM,
    "command_r": ModelType.LLM,
    "internlm": ModelType.LLM,
    "internlm2": ModelType.LLM,
    "deepseek_v2": ModelType.LLM,
    "yi": ModelType.LLM,
    "starcoder2": ModelType.LLM,
    "codegen": ModelType.LLM,
    "olmo": ModelType.LLM,
    "olmo2": ModelType.LLM,
    # TTS
    "bark": ModelType.TTS,
    "speecht5": ModelType.TTS,
    "vits": ModelType.TTS,
    "fastspeech2_conformer": ModelType.TTS,
    "seamless_m4t": ModelType.TTS,
    "mms": ModelType.TTS,
    "parler_tts": ModelType.TTS,
    # STT
    "whisper": ModelType.STT,
    "wav2vec2": ModelType.STT,
    "hubert": ModelType.STT,
    "speech_to_text": ModelType.STT,
    "wavlm": ModelType.STT,
    "data2vec-audio": ModelType.STT,
    # Image
    "stable-diffusion": ModelType.IMAGE_GEN,
    "sdxl": ModelType.IMAGE_GEN,
    "vit": ModelType.IMAGE_CLASS,
    "resnet": ModelType.IMAGE_CLASS,
    "convnext": ModelType.IMAGE_CLASS,
    "swin": ModelType.IMAGE_CLASS,
    "detr": ModelType.OBJECT_DETECT,
    "yolos": ModelType.OBJECT_DETECT,
    "sam": ModelType.IMAGE_SEG,
    "maskformer": ModelType.IMAGE_SEG,
    "segformer": ModelType.IMAGE_SEG,
    # Embeddings / MLM
    "bert": ModelType.EMBEDDING,
    "roberta": ModelType.EMBEDDING,
    "distilbert": ModelType.EMBEDDING,
    "xlm-roberta": ModelType.EMBEDDING,
    "mpnet": ModelType.EMBEDDING,
    "deberta": ModelType.EMBEDDING,
    "deberta-v2": ModelType.EMBEDDING,
    "clip": ModelType.EMBEDDING,
    "siglip": ModelType.EMBEDDING,
    # Seq2Seq
    "t5": ModelType.SUMMARIZATION,
    "mt5": ModelType.SUMMARIZATION,
    "bart": ModelType.SUMMARIZATION,
    "mbart": ModelType.TRANSLATION,
    "pegasus": ModelType.SUMMARIZATION,
    "marian": ModelType.TRANSLATION,
    "m2m_100": ModelType.TRANSLATION,
    "nllb": ModelType.TRANSLATION,
    # Visual QA / Multimodal
    "blip": ModelType.VISUAL_QA,
    "blip-2": ModelType.VISUAL_QA,
    "llava": ModelType.VISUAL_QA,
    "llava_next": ModelType.VISUAL_QA,
    "qwen2_vl": ModelType.VISUAL_QA,
    "paligemma": ModelType.VISUAL_QA,
    "idefics2": ModelType.VISUAL_QA,
    "idefics3": ModelType.VISUAL_QA,
    "mllama": ModelType.VISUAL_QA,
    "florence2": ModelType.IMAGE_TEXT,
    # Video
    "video_mae": ModelType.VIDEO,
    "timesformer": ModelType.VIDEO,
    "cogvideox": ModelType.VIDEO,
    # Depth
    "dpt": ModelType.DEPTH,
    "depth_anything": ModelType.DEPTH,
    "zoedepth": ModelType.DEPTH,
}


def _check_architecture_match(architectures: list[str]) -> ModelType | None:
    """Check if any architecture matches a known type."""
    for arch in architectures:
        if arch in LLM_ARCHITECTURES:
            return ModelType.LLM
        if arch in TTS_ARCHITECTURES:
            return ModelType.TTS
        if arch in STT_ARCHITECTURES:
            return ModelType.STT
        if arch in IMAGE_GEN_ARCHITECTURES:
            return ModelType.IMAGE_GEN
        if arch in IMAGE_CLASS_ARCHITECTURES:
            return ModelType.IMAGE_CLASS
        if arch in OBJECT_DETECT_ARCHITECTURES:
            return ModelType.OBJECT_DETECT
        if arch in IMAGE_SEG_ARCHITECTURES:
            return ModelType.IMAGE_SEG
        if arch in EMBEDDING_ARCHITECTURES:
            return ModelType.EMBEDDING
        if arch in FILL_MASK_ARCHITECTURES:
            return ModelType.FILL_MASK
        if arch in TOKEN_CLASS_ARCHITECTURES:
            return ModelType.TOKEN_CLASS
        if arch in QA_ARCHITECTURES:
            return ModelType.QA
        if arch in SEQ2SEQ_ARCHITECTURES:
            return ModelType.SUMMARIZATION
        if arch in VISUAL_QA_ARCHITECTURES:
            return ModelType.VISUAL_QA
        if arch in IMAGE_TEXT_ARCHITECTURES:
            return ModelType.IMAGE_TEXT
        if arch in DEPTH_ARCHITECTURES:
            return ModelType.DEPTH
        if arch in AUDIO_CLASS_ARCHITECTURES:
            return ModelType.AUDIO_CLASS
        if arch in VIDEO_ARCHITECTURES:
            return ModelType.VIDEO
    return None


def detect_model_type(model_path: Path) -> ModelType:
    """Detects the type of model based on config.json.

    Checks architectures, model_type field, and pipeline_tag to determine
    the model's functionality (LLM, TTS, STT, image generation, etc.).

    Args:
        model_path: Path to the model directory or file

    Returns:
        ModelType indicating the model's functionality
    """
    import json

    # If it's a file, get the parent directory
    if model_path.is_file():
        model_path = model_path.parent

    config_path = model_path / "config.json"
    if not config_path.exists():
        # Check for GGUF files - these are always LLM
        if any(model_path.glob("*.gguf")):
            return ModelType.LLM
        return ModelType.UNKNOWN

    try:
        with open(config_path) as f:
            config = json.load(f)

        # 1. Check architectures field (most reliable)
        architectures = config.get("architectures", [])
        arch_type = _check_architecture_match(architectures)
        if arch_type is not None:
            return arch_type

        # 2. Check model_type field
        model_type_field = config.get("model_type", "").lower().replace("-", "_")
        if model_type_field in MODEL_TYPE_FIELD_TO_TYPE:
            return MODEL_TYPE_FIELD_TO_TYPE[model_type_field]

        # 2b. Pattern matching for model_type field (catches brand-specific types)
        if model_type_field:
            # TTS patterns: *_tts, *tts, speech*, etc.
            if model_type_field.endswith("_tts") or model_type_field.endswith("tts"):
                return ModelType.TTS
            if "speech" in model_type_field and "to_text" not in model_type_field:
                return ModelType.TTS
            # STT patterns: *_asr, *_stt, *speech_to_text*, whisper*, etc.
            if model_type_field.endswith("_asr") or model_type_field.endswith("_stt"):
                return ModelType.STT
            if "speech_to_text" in model_type_field or "asr" in model_type_field:
                return ModelType.STT

        # 3. Check pipeline_tag (if present, usually from model card)
        pipeline_tag = config.get("pipeline_tag", "")
        if pipeline_tag in PIPELINE_TAG_TO_TYPE:
            return PIPELINE_TAG_TO_TYPE[pipeline_tag]

        # 4. Check for diffusers models (look for scheduler config)
        if (model_path / "scheduler").exists() or config.get("_class_name", "").endswith(
            "Pipeline"
        ):
            return ModelType.IMAGE_GEN

        # 5. Check for sentence-transformers
        if (model_path / "sentence_bert_config.json").exists():
            return ModelType.EMBEDDING

        # 6. Default: if it has weights, assume LLM (most common)
        if any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin")):
            # Check if it looks like an encoder-only model (BERT-like)
            if config.get("is_encoder_decoder") is False and not any(
                "ForCausalLM" in a or "ForConditionalGeneration" in a for a in architectures
            ):
                return ModelType.EMBEDDING
            return ModelType.LLM

        return ModelType.UNKNOWN

    except (json.JSONDecodeError, OSError):
        return ModelType.UNKNOWN
