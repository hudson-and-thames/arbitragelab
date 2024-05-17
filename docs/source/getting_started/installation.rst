.. _getting_started-installation:

============
Installation
============

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

      pip install arbitragelab

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

    .. note::

        If you are using on Apple Silicon, and Python 3.8 or 3.9, first install the ``cvxpy``
        dependency before installing ArbitrageLab, as follows:

        .. code-block::

            conda install -c conda-forge cvxpy


    .. code-block::

        pip install arbitragelab

#. You are now ready to use ArbitrageLab.


Google Colab
############

.. note::

   Google Colab frequently updates the version of Python it used. You might need to
   explore additional methods of using a Python 3.8 kernel for full support of
   ArbitrageLab.

#. Open a new Terminal, and install ArbitrageLab using ``pip``:

   .. code-block::

      pip install arbitragelab
