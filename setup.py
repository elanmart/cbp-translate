from setuptools import find_packages, setup

setup(
    name="cbp-translate",
    version="0.0.1",
    url="https://github.com/elanmart/cyberpunk-translator.git",
    author="Marcin Elantkowski",
    author_email="marcin.elantkowski@gmail.com",
    packages=find_packages(
        include=["cbp_translate", "cbp_translate.*"], exclude=["playground"]
    ),
)
