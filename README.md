<p align="center"><img src="docs/art.png" alt="image" width="200" height="auto"></p>

# ART - Actually Robust Training framework

![Tests](https://github.com/SebChw/art/actions/workflows/tests.yml/badge.svg)
![Docs](
https://readthedocs.org/projects/actually-robust-training/badge/?version=latest&style=flat)

----

**ART** is a framework that teaches and keeps an eye on good practices when training deep neural networks. It is inspired by a [blog post by Andrej Karpathy “A Recipe for Training Neural Networks”](https://karpathy.github.io/2019/04/25/recipe/). The framework teaches the user how to properly train DNNs by encouraging the user to use built-in mechanisms that ensure the correctness and robustness of the pipeline using easily usable steps. It allows users not only to learn but also to use it in their future projects to speed up model development.

**Table of contents:**
- [ART - Actually Robust Training framework](#art---actually-robust-training-framework)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
    - [Steps](#steps)
    - [Adding checks](#adding-checks)
    - [Debug your Neural Network](#debug-your-neural-network)
    - [Get control over what is being calculated](#get-control-over-what-is-being-calculated)
  - [Project creation](#project-creation)
  - [Dashboard](#dashboard)
  - [Tutorials](#tutorials)
  - [Required knowledge](#required-knowledge)
  - [Contributing](#contributing)

## Installation
To get started, install ART package using:
```sh
pip install art-training
```

## Quickstart

### Steps
You can split your entire pipeline into series of steps
```python
    from art.project import ArtProject
    from art.steps import CheckLossOnInit, Overfit, OverfitOneBatch
    
    datamodule = YourLightningDataModule()
    model_class = YourLightningModule

    project = ArtProject(name="quickstart", datamodule=datamodule)

    project.add_step(CheckLossOnInit(model_class))
    project.add_step(OverfitOneBatch(model_class))
    project.add_step(Overfit(model_class))

    project.run_all()
```

As a result, if you logged any metrics with self.log, you will observe something like this:

```
Summary:
Step: Check Loss On Init, Model: YourLightningModule, Passed: True. Results:
         loss-valid: 16.067102432250977
         accuracy-train: 0.75
Step: Overfit One Batch, Model: YourLightningModule, Passed: True. Results:
         loss-train: 0.00035009183920919895
         accuracy-train: 0.800000011920929
Step: Overfit, Model: YourLightningModule, Passed: True. Results:
         loss-train: 2.1859991550445557
         accuracy-train: 0.75
         loss-valid: 3.5187878608703613
```

[Explore all available Steps](not-existing-link)

### Adding checks
Additionally you can verify if step is passed. You can't move forward withouth passing previous steps.

```python
    from art.steps import CheckLossOnInit, Overfit, OverfitOneBatch

    datamodule = YourLightningDataModule()
    model_class = YourLightningModule

    project = ArtProject(name="quickstart", datamodule=datamodule)

    # Let's assume we work on typical classification_problem
    NUM_CLASSES = 10
    EXPECTED_LOSS = -math.log(1 / NUM_CLASSES) # 2.3

    project.add_step(
        CheckLossOnInit(model_class),
        #You must know logged metric names
        checks=[CheckScoreCloseTo("loss-valid", EXPECTED_LOSS, rel_tol=0.01)],
    )
    project.add_step(
        OverfitOneBatch(model_class),
        checks=[CheckScoreLessThan("loss-train", 0.001)],
    )
    project.add_step(
        Overfit(model_class),
        checks=[CheckScoreLessThan("accuracy-train", 0.9)],
    )

    project.run_all()
```

This time, depending on your scores you may observe something like this

```
Check failed for step: Check Loss On Init. Reason: Score 16.067102432250977 is not equal to 2.3
Summary:
Step: Check Loss On Init, Model: YourLightningModule, Passed: False. Results:
         loss-valid: 16.067102432250977
         accuracy-train: 0.75
```

### Debug your Neural Network
Track network evolution, gradients values, save images by Decorating your functions
```python
    from art.decorators import BatchSaver, art_decorate

    datamodule = YourLightningDataModule()
    model_class = YourLightningModule
    art_decorate([(model_class, "forward")], BatchSaver())

    project = ArtProject(name="quickstart", datamodule=datamodule)
    # For this to work we should allow providing empty checks. And just print warning... man use checks.
    project.add_step(CheckLossOnInit(model_class))
    project.run_all()

```

### Get control over what is being calculated
For some problems metrics calculation can be quite expensive. By utilizing `MetricCalculator` we have control over what is being calculated. Additionally, metric names and logging is handled automatically
```python
    from art.metrics import MetricCalculator, SkippedMetric

    datamodule = YourLightningDataModule()
    # from art.core import ArtModule
    # from art.utils.enums import PREDICTION, TARGET
    # YourLightningModule(L.LightningModule) -> YourArtModule(ArtModule)
    #
    # Now you can Use YourArtModule.compute_metrics({PREDICTION: ..., TARGET: ...}) -> Dict[str, float]
    # Which will calculate and log metrics with appropriate name for you. Additionally you can now skip expensive metrics calculations
    model_class = YourArtModule

    # Let's assume we work on typical classification_problem
    NUM_CLASSES = 10
    EXPECTED_LOSS = -math.log(1 / NUM_CLASSES)

    # Remove metrics definition from model and put them here
    expensive_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    project = ArtProject(name="quickstart", datamodule=datamodule)

    metric_calculator = MetricCalculator(project, [expensive_metric, loss_fn])

    project.add_step(
        CheckLossOnInit(model_class),
        # You don't have to hardcode this names
        checks=[CheckScoreCloseTo(loss_fn, EXPECTED_LOSS, rel_tol=0.01)],
        # expensive_metric won't be calculated during this step
        skipped_metrics=[SkippedMetric(expensive_metric)],
    )
    project.add_step(
        OverfitOneBatch(model_class),
        checks=[CheckScoreLessThan(loss_fn, 0.1)],
        skipped_metrics=[SkippedMetric(expensive_metric)],
    )
    project.add_step(
        Overfit(model_class),
        checks=[CheckScoreGreaterThan(expensive_metric, 0.9)],
    )

    project.run_all(metric_calculator=metric_calculator)
```

If you want to use all features from ART, and create your new Deep Learning Project following good practices consider checking [tutorials](#tutorials). They cover more features and bigger tasks

## Project creation
To use most of art's features we encourage you to create a new folder for your project using the CLI tool:
```sh
python -m art.cli create-project my_project_name
```

This will create a new folder `my_project` with a basic structure for your project. To learn more about ART we encourage you to read our [documentation](https://actually-robust-training.readthedocs.io/en/latest/), and check our [tutorials](#tutorials)!

## Dashboard
After you run some steps you can see compare their execution in the dashboard. To use the dashboard, firstly install required dependencies:
```sh
pip install art-training[dashboard]
```
and run this command in the directory of your project (directory with folder called art_checkpoints).
```sh
python -m art.cli run-dashboard
```
Optionally you can use --experiment-folder switch to pass path to the folder. For more info, use --help switch.

## Tutorials
1. A showcase of ART's features. To check it out type:
```sh
python -m art.cli get-started
```
and launch tutorial.ipynb

After running all cells run dashboard with

```sh
python -m art.cli run-dashboard
```

2. A tutorial showing how to use ART for transfer learning in an NLP task.
```sh
python -m art.cli bert-transfer-learning-tutorial
```


## Required knowledge
In order to use ART, you should have a basic knowledge of:
- Python - you can find many tutorials online, e.g. [here](https://www.learnpython.org/)
- Basic knowledge of machine learning & neural networks - you can find many tutorials online, e.g. [here](https://www.coursera.org/learn/machine-learning)
- PyTorch - you can find many tutorials online, e.g. [here](https://pytorch.org/tutorials/)
- PyTorch Lightning - you can find many tutorials online, e.g. [here](https://lightning.ai/docs/pytorch/stable/levels/core_skills.html)

## Contributing
We welcome contributions to ART! Please check out our [contributing guide](https://github.com/SebChw/art/wiki/Contributing)