from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="emosense",
    version="0.1.0",
    description="EmoSENSE: sentiment-semantic fuzzy hierarchical RL for emotional image generation",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Junyi Guo",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch<2.5",
        "torchvision",
        "transformers>=4.45.2",
        "diffusers>=0.30.3",
        "accelerate>=0.26.1",
        "timm",
        "peft>=0.9.0",
        "safetensors",
        "Pillow",
        "numpy",
        "tqdm",
        "modelscope",
        "setuptools",
    ],
)
