
`NA POCZATKU ABSTRAKCYJNIE!!! DEFINIUJEMY TYLKO PRZEPIS DLA UZYTKOWNIKA A ON NA RAZIE WSZYSTKO MUSI ZROBIC SAM. JAK TO SIE UDA TO BEDZIEMY ZA NIEGO PROBOWAC AUTOMATYZOWAC` 



1. We need to abstract things! Most of the code seems to be project specific -> everyone agree
2. Configs shouldn't be by default in the art library. We should create some automatic cookiecutter etc. So that if someone needs some predefined config one downloads it on the fly. To many configs -> mess. 
   1. Kacper - configi should be in art and just be copied as user issues `python -m art.create_project`. Try `cookiecooter`.
3. We should create same template for other files. I see it that way someone downloads art and run something like `python -m art.create_project` and all cells with project specific codes should be generated. Additionally for benchmarks like MNIST etc. we should have such templates with code already filled in prepared!. 
   1. In art we have BaseClass. In local folders we copy some file with inherited class. 
4. Modification of these configs is very cumbersome. Maybe we should write a simple app that would allow doing so?
   1. `use hydra and change things using --attribute`

`For now we should be building rather mocks and try to bundle them together well`
crucial stuff to be done for me:
* Create all Stages classes and try to somehow wrap them into the experiment class.
* Create template preparation scripts. 
  * `POC` on how to do this best. See `React` or `Node` or `poetry`, `spacy` (automatyczny kreator configow w spacy)
* Do research how users can inject custom behaviours! and prepare some POC this is really crucial as this is the main part of art.
* Do research on how to handle loggers (supporting many should be very good) and prepare POC. We want to have funcionalities like training continuation, uploading images etc.
* create POC of the system for visualizations. If we do this well this potentially can be big advantage. I see it in the way that we have an object (Visualizaer) that takes as attrbiute hugging face dataset and we can assign some callbacks to it that later can be used inside dash app?



`CLI`, `core` and `dashboard` is separated from everything.

We have `Project` class that is aware of every experiment


`Project` consits of `Experiments` that consist of `Stages` each Stage has check.

In `dashboard`:
* We have 1 model and we want to train it as well as possible
* We have many models and we want to see which works best (`I think this is the best option`)


`Don't focus on dashboard for now` create CLI.

BaseComponents a nastepnie uzytkownik je wszystkie rozszerza.

Visualizers:
* chcemy zwizualizowac segmentacje. Wizualizer robi to i zapisuje w sobie. To moze byc wykorzystane pozniej przez inny komponent. Wizualizer moze zrobic to w dowolnym momencie. Wszystkie obiekty maja dostep do jednego wizualizera i biora z niego co chca.

* Osobny visualizer dla eksploracji danych. Polaczyc to z `DataAnalyzer`

Ostatni stage -  Proste stworzenie API z dockerem. `fastapi`, `gremio`


* `SPROBOWAC Pomyslec jak podpiac pod to GPT`. Na razie bez zadnego contekstu. Na faze testowanie Uzyc jakiegos Mocka.

* Pododawac jakies hinty do poszczegolnych stage. Na razie Zmockowane, jakies bardzo ogolne a potem ewentualnie customowe.

* na pozniej `POC` jak polaczyc DL modele i ML modele na etapie baselinu.