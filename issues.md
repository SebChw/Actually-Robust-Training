09.07.2023


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

* create POC of the system for visualizations. If we do this well this potentially can be big advantage. I see it in the way that we have an object (Visualizaer) that takes as attrbiute hugging face dataset and we can assign some callbacks to it that later can be used inside dash app?

* Should our checks be in tests folder?
NO

* Visualizer should be part o f step?
It will go out IN LAUNDRY

* Visualizer should be responsible only for flot/table/log message generation. Rest should be handled by gui.

* Do you think that we would need experiment level checks? I think only step level checks are enough.


====================================================================================================================
18.07.2023
`Mateusz TODO`: Logger, co potrzebujemy:
* Mozliwosc wrzucania obrazkow z visualizera itp. - Must have
* Mozliwosc kontynuacji treningu z checkpointu zapisanego na serwisie logujacym - mniej wazne.
* Logowanie configu - Must have

Problemy z konfliktem w configu:
* Jesli pobieramy configa z neta, to robimy tak jak git commit bez flagi `m` i komus w edytorze wyswietla sie config, ktory moze zmienic albo zaakceptowac. Sciagasz config z batch_size 16 a Twoja maszyna pomiesci np 8 itp. Ustawienia pathow itd.
  * Rozdzielmy configi dotyczace ustawien systemowych od configow ktore sie nie zmienia.

Przygotowywanie templatek:
* powinnismy inspirowac sie spaCy
* klonowanie repo (tak jak spacy) - inteligente, ze bierzemy tylko konkretne pliki.
* Idea zeby pozwolic uzytkownikowi dynamicznie dodawac jakies komponenty, zeby nie zawalic go od razu wszystkim. (sprawdzic jak ciezkie to bedzie). Na samym poczatku totalnie minimum i dynamiczne dodawanie.
* Na nastepny raz jakis PoC.
* check hydra for configs. Try to combine it with SpaCy.
* Karol TODO: `Na razie niech komus sie pojawi templates i folder z configami (hydra) lub jeden wiekszy. Stworz z tym opsobne repo. Czy da sie wybierac z tego folderu pojedyncze pliku`

Struktura projektu:
* Builder do experymentu - to nie jest najlepszy pomysl (chyba)
* Sam step ma w sobie next_step - taki dekorator, ze kazdy step dekorujemy kolejnym. Ale wtedy stepy wiedza o sobie to jest problem. Zly pomysl.
* Step mogly by byc tworzone z configow, hydra potrafi inicjalizowac obiekty
* Na razie zostanmy przy liscie zrobimy to potem.
* Stepy nie komunikuja sie z soba przez pythona tylko przez zapisywane pliki.
* W jaki sposob stage informuja o tym, gdzie zapisal swoj wynik. Umozliwienie komunikacji pomiedzy stepami moze miec sens.

Kacper, Sebastian: `TODO: Sprobowac odpalic wszystkie stepy po kolei z dummy projektem.`

======================================================================================================================

25.07.2023

Issues Sebastian:
* jak wprowadzac regularyzacje modelu i zbioru danych? -  pierwsze pytanie gdzie ma byc ta metoda czy w LightningModule czy gdzies poza. Alternatywa `turn_on_regularization(model, **kwargs)`
* jak rozwiazac calculate metrics, zeby kazdy mial wspolna. Wielokrotne dziedziczenie. Zrobic cos ala `interfejs` i wtedy uzytkownik definiuje tylko ten interfejs (Swietny pomysl).
* Ogarnac, tak zeby ladnie sie zewszad importowalo. Narzucic jakis styl pisania. Napisac instrukcje jak importowac. Nie zostawialbym tego na usera.

Issues Karol:
* Osobne templatki - osobne branche. Na ten moment git
* Raczej nadal bym myslal jak to zrobic lepiej zeby nie skonczyc z 20 branchami.
* Sprawdzic mozliwosci https://cookiecutter.readthedocs.io/en/stable/README.html i byc moze zastapil bym uzywanie `git` wlasnie tym cookiecooterem
* `typer` + `hydra`, cool.
* Sprawdzilbym jak to robi `react`, `poetry`. Wybrac z czego mozemy wziac najlepsze rzeczy. Byc moze przygotowac jakis `raport` z tego.

Issues Mati:
* Na razie MlFlow odpuszczamy.
* Jak wrzucamy obrazki to mozemy wlaczyc optymalizacje ich rozmiarow, zeby nie zawalaly bardzo duzo miejsca.
* Na razie nie personalizujemy, pomysl
* Fabryka Loggerow - rozwazyc.
* Ogarnac to ladowanie z checkpointow - `top priority`. Wrzucam checkpoint potem moge go pobrac i kontynuowac trening
* Ogarnac synchornizacje folderow lokalnie a z wandb/neptune.
  

============================================================================================================================
08.03.2023

W sumie do logerow mozna by uzyc registering decorator, zeby uzywac kilku na raz.

**Na razie robimy tylko jeden typ zadania i spisujemy potencjalne problemy.**

Karol:
* Osobny skrypt, ktory wypelnia config.
* Mamy jedna glowna templatke
* Pozniej dynamicznie tworzymy projekt
* Mamy mozliwosc dodac jakas czesc juz po stworzeniu projektu.
* `cookiecutter` to jest niesamowity kombain mozna zrobic w nim wszystko o czym mowilismy.


Kacper:
* dodac do nazw cyferki zeby bylo wiadomo, kto kiedy 
* Do dopracowania komunikacja pomiedzy krokami
* `Inplace` czy nie `Inplace`. Za inplace jest taki, zeby nie marnowac pamieci na GPU. Przy kazdym stepie inicjalizowac na nowo. ?
* `Dekorator` ktory czysci po kazdym stepie 

Logger:
* Na razie testowac tylko `Neptune`

Sebastian:
* `Compute Metrics` w praktyce.


**Przygotowac jakis tutorial**

Co mamy:
1. Tworzenie projektu.
2. Mozliwosc przejscia po wszystkich krokach.
3. Mozliwosc zapisywania wynikow.
4. Sprawdzanie czy stage przeszedl.

To do:
1. Dodanie puszczenia z CLI + potencjalnie na teraz jakis raport ASCI -> `Karol`
2. Dodanie wizualizacji w czasie wykonywanie stepow, ktore ida do tego samego folderu. -> `Sebastian`
3. Tutorial -> `Sebastian`/`Kacper`
4. LOGO -> `DB`
5. Cel: Brzezinski + tutorial -> Brzezinski z wytrenowanym MNISTem = sukces.
6. Zrobic tak, zeby templates to bylo osobne repo -> `Sebastian`/`Kacper`
7. Add order in checkpoint folders