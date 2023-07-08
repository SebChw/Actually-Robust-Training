:py:mod:`art.utils.neptun_api_wrapper`
======================================

.. py:module:: art.utils.neptun_api_wrapper

.. autodoc2-docstring:: art.utils.neptun_api_wrapper
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NeptuneApiWrapper <art.utils.neptun_api_wrapper.NeptuneApiWrapper>`
     - .. autodoc2-docstring:: art.utils.neptun_api_wrapper.NeptuneApiWrapper
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_last_run <art.utils.neptun_api_wrapper.get_last_run>`
     - .. autodoc2-docstring:: art.utils.neptun_api_wrapper.get_last_run
          :summary:
   * - :py:obj:`push_configuration <art.utils.neptun_api_wrapper.push_configuration>`
     - .. autodoc2-docstring:: art.utils.neptun_api_wrapper.push_configuration
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CONFIG_FILE <art.utils.neptun_api_wrapper.CONFIG_FILE>`
     - .. autodoc2-docstring:: art.utils.neptun_api_wrapper.CONFIG_FILE
          :summary:

API
~~~

.. py:data:: CONFIG_FILE
   :canonical: art.utils.neptun_api_wrapper.CONFIG_FILE
   :value: None

   .. autodoc2-docstring:: art.utils.neptun_api_wrapper.CONFIG_FILE

.. py:class:: NeptuneApiWrapper(project_name, run_id)
   :canonical: art.utils.neptun_api_wrapper.NeptuneApiWrapper

   .. autodoc2-docstring:: art.utils.neptun_api_wrapper.NeptuneApiWrapper

   .. rubric:: Initialization

   .. autodoc2-docstring:: art.utils.neptun_api_wrapper.NeptuneApiWrapper.__init__

   .. py:method:: get_checkpoint(path='./')
      :canonical: art.utils.neptun_api_wrapper.NeptuneApiWrapper.get_checkpoint

      .. autodoc2-docstring:: art.utils.neptun_api_wrapper.NeptuneApiWrapper.get_checkpoint

.. py:function:: get_last_run(cfg)
   :canonical: art.utils.neptun_api_wrapper.get_last_run

   .. autodoc2-docstring:: art.utils.neptun_api_wrapper.get_last_run

.. py:function:: push_configuration(logger, cfg: omegaconf.DictConfig)
   :canonical: art.utils.neptun_api_wrapper.push_configuration

   .. autodoc2-docstring:: art.utils.neptun_api_wrapper.push_configuration
