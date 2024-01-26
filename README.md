![steampunkFOURcrop](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/assets/65875290/df798f4a-cb2c-416f-a150-093b9382a621)

[![Python Versions](https://img.shields.io/pypi/pyversions/animalai)](https://pypi.org/project/animalai/) [![PyPI](https://img.shields.io/pypi/v/animalai)](https://pypi.org/project/animalai/) [![PyPI Downloads](https://img.shields.io/pypi/dm/animalai)](https://pypi.org/project/animalai/)

# Animal-AI

This repository manages the Python interface for the Animal-AI environment.

The main project repository is located [here](https://github.com/Kinds-of-Intelligence-CFI/animal-ai).

* **Website:** [https://www.animalai.org](https://animalai.org/)
* **Documentation:** [https://animalai.org/doc](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/tree/main/docs)
* **Unity Source code:** [https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project](https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity-project)
* **Python Source code:** [https://github.com/Kinds-of-Intelligence-CFI/animal-ai-python](https://github.com/Kinds-of-Intelligence-CFI/animal-ai-python)
* **Bug reports:** [https://github.com/Kinds-of-Intelligence-CFI/animal-ai/issues](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/issues)

For more information about the ways you can contribute to Animal-AI, visit our website. If you’re unsure where to start or how your skills fit in, reach out! You can ask on GitHub, by opening a new issue or leaving a comment on a relevant issue that is already open.

If you are new to contributing to open source, [this](https://github.com/Kinds-of-Intelligence-CFI/animal-ai/blob/main/CONTRIBUTING.md) guide helps explain why, what, and how to successfully get involved.

## Version History

* v4.0.0
  + Only implements _mlagents 0.30.0_ package to avoid dependency issues; also reduces package size considerably.
  + Cleaned up package and removed redundant files.
  + Implemented tests for the package.
  + Added documentation for the package.
**Note: Version 4.0.0 is not backward compatible with previous versions of Animal-AI due to the significance of the changes made.**
* v3.0.5
  + Removed redundant packages in setup.py.
  + Added download stats.
* v3.0.4
  + Added current and past contributors.
  + Added project description and metadata.
* v3.0.3
  + Asserted python version to be 3.6.1 or higher, but less than 3.10.0 (exclusive).
  + Added package description to setup.py and package metadata.
* v3.0.2
  + Fixed major package dependency issues, related to mlagents 0.30.0, protobuf, and shimmy.
  + Updated project setup.py to accommodate the latest version of Animal-AI package dependencies.
    - Users can now use `pip install animalai` to install latest version of Animal-AI from PyPI effortlessly.
