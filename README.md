![steampunkFOURcrop](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/assets/65875290/df798f4a-cb2c-416f-a150-093b9382a621)

[![Python Versions](https://img.shields.io/pypi/pyversions/animalai)](https://pypi.org/project/animalai/) [![PyPI](https://img.shields.io/pypi/v/animalai)](https://pypi.org/project/animalai/) [![PyPI Downloads](https://img.shields.io/pypi/dm/animalai)](https://pypi.org/project/animalai/)

# Animal-AI

This repository manages the Python interface for the Animal-AI environment.

The main project repository is located [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai).

* **Website:** [here](https://sites.google.com/csah.cam.ac.uk/animalai/)
* **Unity Source code:** [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project)
* **Python Source code:** [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai-python)
* **Bug reports:** [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/issues)

For more information about the ways you can contribute to Animal-AI, visit our website. If you’re unsure where to start or how your skills fit in, reach out! You can ask on GitHub, by opening a new issue or leaving a comment on a relevant issue that is already open.

If you are new to contributing to open source, [this](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/blob/main/CONTRIBUTING.md) guide helps explain why, what, and how to successfully get involved.

## Version History
* v5.0.0
**Note: Version 5.0.0 is not backward compatible with previous versions of Animal-AI due to breaking changes.**
  + Minimum Python version is now `3.10.0 (<3.10.13)` (breaking change).
  + Upgraded to ml-agents-ml-env `1.0.0`.
  + Adds no graphics monitor support.
* v4.1.0
  + Updated `RaycastParser` to accept new object:
    - `HollowBox`.
  + Added a new low-level random agent implemented on Braitenberg model.
  + Bug fixes and performance improvements, specifically on improving the reliability of the Braitenberg model.
  + Added built-in functionality to run yaml configuration files directly via Python.
* v4.0.1
  + Updated RaycastParser to accept two new objects:
    - `DecoyGoal` and `DecoyGoalBounce`.
* v4.0.0
**Note: Version 4.0.0 is not backward compatible with previous versions of Animal-AI due to breaking changes.**
  + Only implements _`mlagents 0.30.0`_ package to avoid dependency issues; also reduces package size considerably.
  + Cleaned up package and removed redundant files.
  + Implemented tests for the package.
  + Added documentation for the package.
* v3.0.5
  + Removed redundant packages in `setup.py`.
  + Added download stats.
* v3.0.4
  + Added current and past contributors.
  + Added project description and metadata.
* v3.0.3
  + Asserted python version to be 3.6.1 or higher, but less than 3.10.0 (exclusive).
  + Added package description to `setup.py` and package metadata.
* v3.0.2
  + Fixed major package dependency issues, related to `mlagents 0.30.0`, `protobuf`, and `shimmy`.
  + Updated project `setup.py` to accommodate the latest version of Animal-AI package dependencies.
    - Users can now use `pip install animalai` to install latest version of Animal-AI from PyPI effortlessly.
