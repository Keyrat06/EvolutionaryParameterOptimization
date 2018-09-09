import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["tqdm", "numpy", "matplotlib"]

setuptools.setup(
    name="evolutionary_optimization",
    version="1.0.1",
    author="Raoul Khouri",
    author_email="Raoulemil@gmail.com",
    description="A light weight evolutionary parameter optimization package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Keyrat06/EvolutionaryParameterOptimization",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
