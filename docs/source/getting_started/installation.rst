.. _getting_started-installation:

============
Installation
============


API Keys
########
In order to use ``arbitragelab``, you will require an API key. This is provided to
you when you purchase ArbitrageLab, and can be found on the client portal. If you are unable to find your API key, please reach out to sales@hudsonthames.org.

Recommended Setup on Windows
############################

#. Download and install the latest version of `Anaconda 3 <https://www.anaconda.com/products/individual>`__
#. Launch Anaconda Navigator
#. Click Environments, choose an environment name, select Python 3.8, and click Create
#. Click Home, browse to your new environment, and click Install under Jupyter Notebook
#. Launch the Anaconda Prompt and activate the environment:

   .. code-block::

      conda activate <env_name>

#. Install ArbitrageLab using ``pip``:

   .. code-block::

      pip install https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@raw.githubusercontent.com/hudson-and-thames-clients/arbitragelab/master/arbitragelab-0.8.0-py38-none-any.whl


#. Make sure your API key is available in your environment under the ``ARBLAB_API_KEY`` environment variable by opening the Command Prompt as an administrator, and running the following command:

   .. code-block::

      setx ARBLAB_API_KEY "<your-api-key"

   where you replace ``<your-api-key`` with your ArbitrageLab API key you received
   when you purchased a subscription.

   Verify that your API key is available in your environment by executing the following in a **new** command prompt terminal:

   .. code-block::

      echo %ARBLAB_API_KEY%

   which should echo your API key to the terminal.

#. You are now ready to use ArbitrageLab.


Recommended Setup on Linux / MacOS
##################################

.. note::

   If you are running on Apple Silicon, you will need to make sure `Homebrew
   <https://brew.sh/>`__ is installed, and that you have installed ``cmake``:

   .. code-block::

      brew install cmake



#. Install some variant of ``conda`` environment manager (we recommend Anaconda or Miniconda) for your platform.
#. Launch a new terminal and create a new ``conda`` environment using your environment manager:

   .. code-block::

      conda create -n <env_name> python=3.8

#. Make sure the environment is activated:

   .. code-block::

      conda activate <env_name>

#. Install ArbitrageLab using ``pip``:

   .. code-block::

      pip install https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@raw.githubusercontent.com/hudson-and-thames-clients/arbitragelab/master/arbitragelab-0.8.0-py38-none-any.whl

#. Make sure your API key is available in your environment under the ``ARBLAB_API_KEY`` environment variable.

   If you're running on **Linux or MacOS**, you can add the following to your
   ``~/.zshrc``, ``~/.bashrc`` or ``~/.profile`` file in order to make the API
   key available:

   .. code-block::

      ARBLAB_API_KEY="<your-api-key>"

   where you replace ``<your-api-key`` with your ArbitrageLab API key you received
   when you purchased a subscription.

   .. note::

      Remember to close and open a new terminal, or ``source`` your terminal
      configuration to make sure the environment gets refreshed with our new
      ``ARBLAB_API_KEY`` variable.


   Verify that your API key is available in your environment by executing the following in your refreshed terminal

   .. code-block::

      echo $ARBLAB_API_KEY

   which should echo your API key to the terminal.


#. You are now ready to use ArbitrageLab.


Google Colab
############

.. note::

   Google Colab frequently updates the version of Python it used. You might need to
   explore additional methods of using a Python 3.8 kernel for full support of
   ArbitrageLab. We are currently working on bring Python 3.9+ support.

#. Open a new Terminal, and install ArbitrageLab using ``pip``:

   .. code-block::

      pip install https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@raw.githubusercontent.com/hudson-and-thames-clients/arbitragelab/master/arbitragelab-0.8.0-py38-none-any.whl


#. Insert the following in the first cell of your notebook in order to register your API key

.. code-block:: python

   # Insert this at the start of your script or notebook
   import os

   os.environ["ARBLAB_API_KEY"] = "<your-api-key>"


Alternative ways of adding the API Key
######################################

.. warning::

   The following is not recommended for security reasons, since you run the risk
   of accidentally leaking your API key in your source code by accidentally
   committing it to your version control system. As a result, we highly
   recommended using environment variables instead.

You can also use the ``os`` module in your Python script or Jupyter notebook
to add the ``ARBLAB_API_KEY`` environment variable before importing
``arbitragelab``:

.. code-block:: python

   # Insert this at the start of your script or notebook
   import os

   os.environ["ARBLAB_API_KEY"] = "<your-api-key>"
