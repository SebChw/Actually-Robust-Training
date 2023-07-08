
1. We need to abstract things! Most of the code seems to be project specific
2. Configs shouldn't be by default in the art library. We should create some automatic cookiecutter etc. So that if someone needs some predefined config one downloads it on the fly. To many configs -> mess
3. We should create same template for other files. I see it that way someone downloads art and run something like `python -m art.create_project` and all cells with project specific codes should be generated. Additionally for benchmarks like MNIST etc. we should have such templates with code already filled in prepared!
4. Modification of these configs is very cumbersome. Maybe we should write a simple app that would allow doing so?


`For now we should be building rather mocks and try to bundle them together well`
crucial stuff to be done for me:
* Create all Stages classes and try to somehow wrap them into the experiment class.
* Create template preparation scripts
* Do research how users can inject custom behaviours! and prepare some POC this is really crucial as this is the main part of art.
* Do research on how to handle loggers (supporting many should be very good) and prepare POC. We want to have funcionalities like training continuation, uploading images etc.
* create POC of the system for visualizations. If we do this well this potentially can be big advantage. I see it in the way that we have an object (Visualizaer) that takes as attrbiute hugging face dataset and we can assign some callbacks to it that later can be used inside dash app?