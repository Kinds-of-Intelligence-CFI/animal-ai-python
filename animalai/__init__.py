from importlib.metadata import PackageNotFoundError

from animalai.download import download_binary as download_binary, get_package_version
from animalai.environment import AnimalAIEnvironment as AnimalAIEnvironment
from animalai import arenas as arenas

try:
    __version__ = get_package_version()
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
