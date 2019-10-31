import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='fastknn',
    version='0.0.1',
    author="Francois Valadier",
    author_email="francois.valadier@openvalue.fr",
    description="Easy to use fast kNN",
    long_description="Easy to use fast kNN",
    long_description_content_type="text/markdown",
    url="https://github.com/Fanchouille/fastknn.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pybind11',
        'nmslib',
        'ujson'],
    classifiers=[
        "Development Status :: 2 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "License :: TF1 :: All rights reserved"
    ],
)
