
1. We need to abstract things! Most of the code seems to be project specific
2. Configs shouldn't be by default in the art library. We should create some automatic cookiecutter etc. So that if someone needs some predefined config one downloads it on the fly. To many configs -> mess
3. We should create same template for other files. I see it that way someone downloads art and run something like `python -m art.create_project` and all cells with project specific codes should be generated. Additionally for benchmarks like MNIST etc. we should have such templates with code already filled in prepared!
4. Modification of these configs is very cumbersome. Maybe we should write a simple app that would allow doing so?