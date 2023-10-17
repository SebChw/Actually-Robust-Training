# ART - Actually Robust Training framework

![Tests](https://github.com/SebChw/art/actions/workflows/tests.yml/badge.svg)
![Docs](
https://readthedocs.org/projects/audio-research-toolkit/badge/?version=latest&style=flat)

----

**ART** is a framework that teaches and keeps an eye on good practices when training deep neural networks. It is inspired by a [blog post by Andrej Karpathy “A Recipe for Training Neural Networks”](https://karpathy.github.io/2019/04/25/recipe/). The framework teaches the user how to properly train DNNs by encouraging the user to use built-in mechanisms that ensure the correctness and robustness of the pipeline using easily usable steps. It allows users not only to learn but also to use it in their future projects to speed up model development.

**Table of contents:**
- [Installation](#installation)
- [Project creation](#project-creation)
- [API Cheatsheet & Documentation](#api-cheatsheet--documentation)
- [Implemented Karpathy's steps](#implemented-karpathys-steps)
- [Tutorials](#tutorials)
- [Required knowledge](#required-knowledge)
- [Contributing](#contributing)

## Installation
To get started, first clone the repo and install our package in editable mode:
```sh
git clone https://github.com/SebChw/art.git && cd art
pip install -e .
```
## Project creation
To use most of art's features we encourage you to create a new folder for your project using the CLI tool:
```sh
python -m art.cli create_project --name my_project
```

This will create a new folder `my_project` with a basic structure for your project. To learn more about ART we encourage you to read our [documentation](https://audio-research-toolkit.readthedocs.io/en/latest/), and check our [tutorials](#tutorials)!

## Tutorials
1. A showcase of ART's features. To check it out type:
```sh
python -m art.cli get_started
```
and go to exp1/tutorial.ipynb

## Required knowledge
In order to use ART, you should have a basic knowledge of:
- Python - you can find many tutorials online, e.g. [here](https://www.learnpython.org/)
- Basic knowledge of machine learning & neural networks - you can find many tutorials online, e.g. [here](https://www.coursera.org/learn/machine-learning)
- PyTorch - you can find many tutorials online, e.g. [here](https://pytorch.org/tutorials/)
- PyTorch Lightning - you can find many tutorials online, e.g. [here](https://lightning.ai/docs/pytorch/stable/levels/core_skills.html)

## Contributing
We welcome contributions to ART! Please check out our [contributing guide](https://github.com/SebChw/art/wiki/Contributing)