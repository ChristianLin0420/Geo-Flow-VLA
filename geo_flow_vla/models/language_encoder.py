"""
Language Encoder for Geo-Flow VLA.

Uses CLIP text encoder for instruction embeddings.
Supports frozen or fine-tunable encoding.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class CLIPLanguageEncoder(nn.Module):
    """
    CLIP-based language encoder for instruction embeddings.
    
    Uses OpenAI's CLIP text encoder (frozen by default).
    Output dimension: 768 for ViT-L/14, 512 for ViT-B/32
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name for CLIP
            max_length: Maximum token length
            freeze: Whether to freeze the encoder
            device: Torch device
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self._freeze = freeze
        self.device = device
        
        # Lazy loading to avoid import issues at module load time
        self._model = None
        self._tokenizer = None
        self._output_dim = None
        
    def _load_model(self) -> None:
        """Lazy load the CLIP model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required for CLIP. "
                "Install with: pip install transformers"
            )
        
        logger.info(f"Loading CLIP text encoder: {self.model_name}")
        
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self._model = CLIPTextModel.from_pretrained(self.model_name)
        
        # Get output dimension from model config
        self._output_dim = self._model.config.hidden_size
        
        # Move to device - use stored device or detect from CUDA availability
        if self.device is not None:
            self._model = self._model.to(self.device)
        elif torch.cuda.is_available():
            # Auto-detect: move to CUDA if available and device wasn't explicitly set
            self.device = torch.device("cuda")
            self._model = self._model.to(self.device)
            logger.info(f"Auto-moving CLIP to CUDA (device was not explicitly set)")
        
        if self._freeze:
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        
        logger.info(f"CLIP language encoder loaded (output_dim={self._output_dim})")
    
    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        if self._output_dim is None:
            # Default for ViT-L/14
            if "large" in self.model_name.lower() or "vit-l" in self.model_name.lower():
                return 768
            elif "base" in self.model_name.lower() or "vit-b" in self.model_name.lower():
                return 512
            return 768  # Default
        return self._output_dim
    
    def to(self, device: torch.device) -> "CLIPLanguageEncoder":
        """Move encoder to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return super().to(device)
    
    def forward(self, text: Union[str, List[str]]) -> Tensor:
        """
        Encode text instruction(s) to embedding.
        
        Args:
            text: Single instruction string or list of strings
            
        Returns:
            Text embeddings (B, output_dim)
        """
        self._load_model()
        
        # Handle single string
        if isinstance(text, str):
            text = [text]
        
        # Get device from model
        device = next(self._model.parameters()).device
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        if self._freeze:
            with torch.no_grad():
                outputs = self._model(**inputs)
        else:
            outputs = self._model(**inputs)
        
        # Use pooled output (CLS token representation)
        embeddings = outputs.pooler_output  # (B, output_dim)
        
        return embeddings

    def encode_batch(self, texts: List[str]) -> Tensor:
        """Batch encode for efficiency."""
        return self.forward(texts)


class MockLanguageEncoder(nn.Module):
    """
    Mock language encoder for testing without CLIP.
    
    Returns random embeddings of the correct dimension.
    """
    
    def __init__(
        self,
        output_dim: int = 768,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._output_dim = output_dim
        self.device = device or torch.device("cpu")
        
        # Simple projection for deterministic-ish outputs
        self.projection = nn.Linear(256, output_dim)
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def to(self, device: torch.device) -> "MockLanguageEncoder":
        self.device = device
        return super().to(device)
    
    def forward(self, text: Union[str, List[str]]) -> Tensor:
        """Generate mock embeddings based on text hash."""
        if isinstance(text, str):
            text = [text]
        
        B = len(text)
        
        # Create deterministic embedding based on text content
        embeddings = []
        for t in text:
            # Use hash of text to seed random generator
            seed = hash(t) % (2**32)
            gen = torch.Generator().manual_seed(seed)
            emb = torch.randn(256, generator=gen)
            embeddings.append(emb)
        
        embeddings = torch.stack(embeddings).to(self.device)
        return self.projection(embeddings)


def create_language_encoder(
    model_name: str = "openai/clip-vit-large-patch14",
    use_mock: bool = False,
    freeze: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function to create language encoder.
    
    Args:
        model_name: CLIP model name
        use_mock: Use mock encoder (for testing)
        freeze: Freeze encoder weights
        device: Torch device
        
    Returns:
        Language encoder module
    """
    if use_mock:
        logger.warning("Using mock language encoder (for testing only)")
        return MockLanguageEncoder(output_dim=768, device=device)
    
    return CLIPLanguageEncoder(
        model_name=model_name,
        freeze=freeze,
        device=device,
    )

