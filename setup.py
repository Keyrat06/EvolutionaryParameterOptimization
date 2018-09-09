import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["tqdm >= 4", "numpy >= 1", "matplotlib >= 2"]

setuptools.setup(
    name="evolutionary_optimization",
    version="1.0.6",
    author="Raoul Khouri",
    author_email="Raoulemil@gmail.com",
    description="A light weight evolutionary parameter optimization package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Keyrat06/EvolutionaryParameterOptimization",
    packages=setuptools.find_packages(),
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    platform='any',
    install_requires=requirements,
    zip_safe=False
)
