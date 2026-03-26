from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from animalai.download import download_binary as download_binary
from animalai.environment import AnimalAIEnvironment as AnimalAIEnvironment
from animalai import arenas as arenas

try:
    __version__ = _get_version("animalai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
