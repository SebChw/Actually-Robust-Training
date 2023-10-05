# audio-research-toolkit

Import lightning consistent as `pytorch_lightning`


![Tests](https://github.com/SebChw/audio-research-toolkit/actions/workflows/tests.yml/badge.svg)

* [Art Design](https://docs.google.com/presentation/d/1m_DTeKvJVMBfEhC76eO9nKPLJp1sLvWXfCvefKd0Hc4/edit?usp=sharing)
* [Presentation from Engineers Talks](https://docs.google.com/presentation/d/1qfoywip9xAw04gx54rBc34LjTqpYK3RGMY-TUVV6NC4/edit?usp=sharing)
* [Survey](https://forms.gle/di8NguugL7y5jhkZA) that aims to gather insights and information about the current practices, challenges, and advancements in the field of training deep neural networks.

To get started, at first clone the repo 

```sh
git clone https://github.com/SebChw/art.git
```

Install audio-research-toolkit in editable mode, also you need to download extras for development if you want to contribute see `[dev]`

```sh
pip install -e .[dev]
```

To build documentation
 ```sh
$ cd docs
$ make clean
$ make html
 ```

If you are to push your code please use

```sh
tox
```
to make sure every check passes

In order to use CLI check:
```sh
python -m art.cli --help
```

To create project from template:
```sh
python -m art.cli create_project --name my_project
```