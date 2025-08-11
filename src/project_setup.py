import logging
from pathlib import Path
from typing import Dict,Any, Union, List
from collections.abc import Mapping

import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import (
    ChatOllama,
    ChatLlamaCpp,
)

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.huggingface import (
    ChatHuggingFace, 
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
LIST_CLIENT = {
    "OPEN_AI": ChatOpenAI,
    "GOOGLE_GENAI": ChatGoogleGenerativeAI,
    "OLLAMA": ChatOllama,
    "LLAMA_CPP": ChatLlamaCpp,
    "HF_LOCAL": ChatHuggingFace,
    "GROQ": ChatGroq,
} 

import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

LIST_EMBEDDER = {
    "GOOGLE_GENAI": GoogleGenerativeAIEmbeddings,
    # "OLLAMA": OllamaEmbeddings,
    # "HF_LOCAL": HuggingFaceEmbeddings,
}

def log_value(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        node, result = func(*args, **kwargs)
        logger.info("Returned from %s: %s", node, result)
        return result
    return wrapper

def to_json(input) -> Dict[str, Any]:
    import json
    """
    Deserialize any JSON‐encoded string values in self.raw_input.
    Non‐string values or invalid JSON remain unchanged.

    :return: a dict with the same keys, but parsed Python objects as values
    """
    if isinstance(input, str):
        input = json.loads(input.replace("'", '"'))
        



def setup_logging(log_file: str = "logs/app.log", *, level: int = logging.DEBUG) -> None:
    """Configure logging for the project.

    Parameters
    ----------
    log_file : str
        Destination file for log messages. A ``logs`` directory will be created
        if it does not already exist.
    level : int
        Logging level passed to :func:`logging.basicConfig`.
    """

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.
    Args:
        path (str): Path to the YAML configuration file.
    """
    cfg = yaml.safe_load(open(path))
    return cfg


def load_nested_config(cfg:dict, key_prefix:str) -> Dict:
    dict_key = key_prefix
    yaml_key = f"{key_prefix}_yaml"

    if dict_key in cfg:
        return cfg[dict_key]
    elif yaml_key in cfg:
        return load_config(cfg[yaml_key])[dict_key]
    else:
        raise ValueError(
            f"Missing configuration: expected either '{dict_key}' or '{yaml_key}' in cfg."
        )



def build_clients(
    confs: Union[Mapping[str, Any], List[Mapping[str, Any]]]
) -> List[BaseChatModel]:
    """
    Factory that returns a list of chat‐model instances.

    Parameters
    ----------
    confs : dict or list of dict
        Each dict must contain a 'type' key (e.g. "OPENAI", "LLAMA_CPP", …),
        plus any other kwargs for that model's constructor.

    Returns
    -------
    List[BaseChatModel]
    """
    # normalize to list
    if isinstance(confs, Mapping):
        confs = [confs]

    clients: List[BaseChatModel] = []
    for conf in confs:
        conf_copy = dict(conf)  # avoid mutating the caller's dict
        kind = conf_copy.pop("type").upper()
        if kind not in LIST_CLIENT:
            raise ValueError(
                f"Unknown client type '{kind}'. Available: {list(LIST_CLIENT)}"
            )
        clients.append(LIST_CLIENT[kind](**conf_copy))

    return clients


def build_embedder(conf: Dict[str, Any]) -> Embeddings:
    kind = conf.pop("type").upper()
    if kind not in LIST_EMBEDDER:
        raise ValueError(
            f"Unknown embedder type '{kind}'. "
            f"Available: {list(LIST_EMBEDDER)}"
        )
    return LIST_EMBEDDER[kind](**conf)
