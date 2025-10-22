logger = logging.getLogger(__name__)

def _log_print(*args, **kwargs):
    try:
        msg = ' '.join((str(a) for a in args))
    except Exception:
        msg = ' '.join(map(str, args))
    logger.info(msg)
import logging
'LM‑Fix module: polished for GitHub publishing.'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any
import os

class ModelCreator:
    """
    A class representing a language model with specific configurations.
    It handles loading the model and tokenizer based on predefined attributes.
    """

    def __init__(self, hf_model_id: str, max_length: int, assistant_pattern: Any, predefined_prompt_templates: dict=None, cache_dir: str='/content/drive/MyDrive/HF_MODELS/', use_flash_attention: bool=True, torch_dtype='auto', aeot_id=[128009], padding_side='right') -> None:
        """
        Initializes the Model with the specified configurations.

        Args:
            hf_model_id (str): Hugging Face model identifier.
            max_length (int): Maximum token length for the tokenizer.
            assistant_pattern (Any): Pattern to identify assistant tokens.
            cache_dir (str, optional): Directory to cache models and tokenizers. Defaults to '/content/drive/MyDrive/HF_MODELS/'.
            use_flash_attention (bool, optional): Whether to enable Flash Attention 2. Defaults to True.
        """
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.assistant_pattern = assistant_pattern
        self.cache_dir = cache_dir
        self.use_flash_attention = use_flash_attention
        self.predefined_prompt_templates = predefined_prompt_templates
        self.torch_dtype = torch_dtype
        self.aeot_id = torch.tensor(aeot_id, dtype=torch.long)
        self.padding_side = padding_side

    def load_model_gpu(self) -> AutoModelForCausalLM:
        """
        Loads the Hugging Face model with the specified configurations.

        Returns:
            AutoModelForCausalLM: The loaded language model.
        """
        _log_print('Loading model with Flash Attention 2...' if self.use_flash_attention else 'Loading model...')
        model = AutoModelForCausalLM.from_pretrained(self.hf_model_id, device_map='cuda', trust_remote_code=True, torch_dtype=self.torch_dtype, use_flash_attention_2=self.use_flash_attention, cache_dir=os.environ.get('HF_HOME', self.cache_dir))
        self.model = model
        return model

    def load_model(self) -> AutoModelForCausalLM:
        """
        Loads the Hugging Face model with the specified configurations.

        Returns:
            AutoModelForCausalLM: The loaded language model.
        """
        _log_print('Loading model with Flash Attention 2...' if self.use_flash_attention else 'Loading model...')
        model = AutoModelForCausalLM.from_pretrained(self.hf_model_id, device_map='auto', trust_remote_code=True, torch_dtype=self.torch_dtype, use_flash_attention_2=self.use_flash_attention, cache_dir=os.environ.get('HF_HOME', self.cache_dir))
        self.model = model
        return model

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Loads and configures the tokenizer for the specified model.

        Returns:
            AutoTokenizer: The configured tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, cache_dir=os.environ.get('HF_HOME', self.cache_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.model_max_length = self.max_length
        tokenizer.max_length = self.max_length
        tokenizer.padding_side = self.padding_side
        self.tokenizer = tokenizer
        return tokenizer

    def get_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Retrieves the loaded model and tokenizer.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and tokenizer.
        """
        return (self.model, self.tokenizer)

    def update_assistant_pattern(self, new_pattern: Any) -> None:
        """
        Updates the assistant pattern used for identifying assistant tokens.

        Args:
            new_pattern (Any): The new assistant pattern.
        """
        self.assistant_pattern = new_pattern
        _log_print(f'Assistant pattern updated to: {self.assistant_pattern}')

    def set_max_length(self, new_max_length: int) -> None:
        """
        Updates the maximum token length for the tokenizer.

        Args:
            new_max_length (int): The new maximum token length.
        """
        self.max_length = new_max_length
        self.tokenizer.model_max_length = new_max_length
        self.tokenizer.max_length = new_max_length
        _log_print(f'Tokenizer max_length set to: {self.max_length}')