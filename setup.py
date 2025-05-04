from setuptools import setup, find_packages

setup(
    name="tiny_modular_transformer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "streamlit>=1.20.0",
        "numpy>=1.20",
        "matplotlib>=3.3"
    ],
    author="ConversionPsychology",
    description="A tiny, modular transformer implementation for education and experimentation.",
    keywords="transformer AI deep-learning pytorch modular",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
