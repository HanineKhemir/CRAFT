from setuptools import setup, find_packages

setup(
    name="mvst_bts",
    version="0.1.0",
    description="MVST-BTS+: Multi-View Spectrogram Transformer with BTS metadata fusion for ICBHI respiratory sound classification",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "scipy>=1.11.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
    ],
)