from typing import List, Union, Any
import tiktoken
import yaml
from datetime import datetime as dt
import attr
from attr import define, field
import sys
sys.path.append("src/")

from helpers.utils import get_api_key




@define
class ChatModel:
    """
    Basic LLM API model wrapper class
    """
    model_provider: str = field(default='azure')
    model_name: str = field(default='gpt-3.5-turbo')
    model_key: Any = field(default=None)
    model_key_path: Any = field(default='secrets.json')
    model_key_name: Any = field(default="api_key_private")
    model: Any = field(default=None)
    temperature: float = 0.0
    context_max_tokens: int = 256

    def __attrs_post_init__(self):

        if self.model_key is None:
            self.model_key = get_api_key(fname=self.model_key_path, provider=self.model_provider,
                                         key=self.model_key_name)

       

    def __call__(self, messages):
        """
        Generate tokens.
        """

        response = self._generate(messages)

        return response


    def history(self):
        # optionally keep a history of interactions
        return self.messages





