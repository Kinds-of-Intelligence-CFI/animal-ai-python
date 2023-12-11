from setuptools import setup, find_packages

setup(
    name="animalai",
    version="3.0.6",
    description="Animal AI Python API",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Kinds-of-Intelligence-CFI/animalai-package",
    author="Matt Crosby; Ibrahim Alhas; Konstantinos Voudouris; Wout Schellaert; Joel Holmes; Ben Beyret",
    author_email="kindsofintelligence.cfi@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    project_urls={
        "Website": "https://sites.google.com/csah.cam.ac.uk/animalai",
        "Main Repository": "https://github.com/Kinds-of-Intelligence-CFI/animal-ai",
        "Releases": "https://github.com/Kinds-of-Intelligence-CFI/animal-ai/releases",
        "Documentation": "https://github.com/Kinds-of-Intelligence-CFI/animal-ai/tree/main/docs",
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    zip_safe=False,
    install_requires=[
        "mlagents==0.30.0",
        "protobuf==3.20.3",
        "numpy==1.21.2",
    ],
    # For OpenAI Gym(nasium) environments. Stable-Baselines3 (SB3) has transitioned to using Gymnasium internally. Requires Shimmy. Currently an optional and can be used like so: pip install animalai[shimmy]'.
    extras_require={
        'shimmy': ['shimmy==1.3.0'],
    },
    python_requires=">=3.6, <3.10",
)
