:py:mod:`art.data.dummy_loader`
===============================

.. py:module:: art.data.dummy_loader

.. autodoc2-docstring:: art.data.dummy_loader
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SanityCheckDataModule <art.data.dummy_loader.SanityCheckDataModule>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`dummy_classification_sample <art.data.dummy_loader.dummy_classification_sample>`
     - .. autodoc2-docstring:: art.data.dummy_loader.dummy_classification_sample
          :summary:
   * - :py:obj:`dummy_source_separation_sample <art.data.dummy_loader.dummy_source_separation_sample>`
     - .. autodoc2-docstring:: art.data.dummy_loader.dummy_source_separation_sample
          :summary:
   * - :py:obj:`dummy_generator <art.data.dummy_loader.dummy_generator>`
     - .. autodoc2-docstring:: art.data.dummy_loader.dummy_generator
          :summary:

API
~~~

.. py:function:: dummy_classification_sample(shape=(1, 16000), label=0)
   :canonical: art.data.dummy_loader.dummy_classification_sample

   .. autodoc2-docstring:: art.data.dummy_loader.dummy_classification_sample

.. py:function:: dummy_source_separation_sample(shape=(2, 44100), instruments=['bass', 'drums', 'vocals', 'other'])
   :canonical: art.data.dummy_loader.dummy_source_separation_sample

   .. autodoc2-docstring:: art.data.dummy_loader.dummy_source_separation_sample

.. py:function:: dummy_generator(sample_gen: typing.Callable, n_samples=32)
   :canonical: art.data.dummy_loader.dummy_generator

   .. autodoc2-docstring:: art.data.dummy_loader.dummy_generator

.. py:class:: SanityCheckDataModule(dataset_generator, collate, num_workers=0)
   :canonical: art.data.dummy_loader.SanityCheckDataModule

   Bases: :py:obj:`lightning.LightningDataModule`

   .. py:method:: prepare_data()
      :canonical: art.data.dummy_loader.SanityCheckDataModule.prepare_data

   .. py:method:: setup(stage)
      :canonical: art.data.dummy_loader.SanityCheckDataModule.setup

   .. py:method:: _get_loader()
      :canonical: art.data.dummy_loader.SanityCheckDataModule._get_loader

      .. autodoc2-docstring:: art.data.dummy_loader.SanityCheckDataModule._get_loader

   .. py:method:: train_dataloader()
      :canonical: art.data.dummy_loader.SanityCheckDataModule.train_dataloader

   .. py:method:: val_dataloader()
      :canonical: art.data.dummy_loader.SanityCheckDataModule.val_dataloader

   .. py:method:: test_dataloader()
      :canonical: art.data.dummy_loader.SanityCheckDataModule.test_dataloader
