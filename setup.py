from setuptools import setup, find_packages

setup(
    name="animalai",
    version="3.0.2",
    description="Animal AI 3 Python API",
    url="https://github.com/Kinds-of-Intelligence-CFI/animalai-package",
    author="Matt Crosby; Ibrahim Alhas; K. Voudouris; W. Schellaert",
    author_email="kindsofintelligence.cfi@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    zip_safe=False,
    install_requires=["mlagents==0.30.0",
                      "numpy==1.21.2",
                      "scipy==1.7.2",
                      "pandas== 1.3.2",
                      "protobuf== 3.20.3",
                      "stable-baselines3== 2.1.0",
                      "shimmy== 1.3.0",
                      "notebook"],
    python_requires=">=3.9, <3.10",
)
