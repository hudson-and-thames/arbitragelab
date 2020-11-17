
============
Installation
============

Recommended Versions
####################

* Anaconda
* Python 3.7 and up.

Installation
############

Mac OS X and Ubuntu Linux
*************************

1. Make sure you install the latest version of the Anaconda distribution. To do this you can follow the install and update instructions found on this `link <https://www.anaconda.com/download/#mac>`_
2. Launch a terminal
3. Create a New Conda Environment. From terminal.

.. code-block::

   conda create -n <env name> python=3.7 anaconda

Accept all the requests to install.

4. Now activate the environment with:

.. code-block::

   source activate <env name>

5. Purchase ArbitrageLab from the `Hudson & Thames website <https://app.hudsonthames.org/auth/signin>`__. This will provide you with an API key.

.. code-block::

    Example: "26303adb02cb759b2d484233162a0"

6. Add API key as an environment variable:

   6.1 The Best Way:

      By adding the API key as an environment variable, you won't need to constantly add the key every time you import the library.

      * Open the terminal and run: ``sudo gedit /etc/environment``
      * This will open a text editor.
      * Add the following environment variable: ``ARBLAB_API_KEY="26303adb02cb759b2d484233162a0"``
      * Note that you must add your own API key and not the one given in this example.
      * Save the file and Logout or restart your computer. (If you skip this step, it won't register the change)

   6.2 The Easy Way:

      If you don't want the key to persist on your local machine, you can always declare each time, before you import the
      ArbitrageLab library.

      * In your python script or notebook, add the following line before you import ArbitrageLab:

      .. code::

         import os
         os.environ['ARBLAB_API_KEY'] = "426303b02cb7475984b2d484319062a0"

7. Install the arbitragelab library into your python environment.

   7.1 From Terminal: install the arbitragelab package:

.. code-block::

   pip install arbitragelab

6. Register your API Key to Unlock the library.



Windows
*******

.. warning::

    Before installing MlFinLab on Windows machines you should download and install
    `Visual Studio build tools for Python3 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_.
    You can use this `installation guide <https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view?usp=sharing>`_.

**Option A (with user interface)**

1. Download and install the latest version of `Anaconda 3 <https://www.anaconda.com/distribution/#download-section>`__
2. Launch Anaconda Navigator
3. Click Environments, choose an environment name, select Python 3.7, and click Create
4. Click Home, browse to your new environment, and click Install under Jupyter Notebook
5. Launch Anaconda Prompt and activate the environment:

.. code-block::

   conda activate <env name>

6. From Anaconda Prompt: install the MlFinLab package:

.. code-block::

   pip install arbitragelab

**Option B (with command prompt)**

1. Download and install the latest version of `Anaconda 3 <https://www.anaconda.com/distribution/#download-section>`__
2. Launch Anacoda Prompt
3. Create new environment (replace <env name> with a name, for example ``arbitragelab``):

.. code-block::

   conda create -n <env name> python=3.7

4. Activate the new environment:

.. code-block::

   conda activate <env name>

5. Install scs library (try one of the below options):

.. code-block::

   pip install scs

.. code-block::

   conda install -c conda-forge scs

.. code-block::

   conda install -c anaconda ecos

6. Install MlFinLab:

.. code-block::

   pip install mlfinlab

.. Note::

    If you have problems with installation related to Numba and llvmlight, `this solution <https://github.com/hudson-and-thames/mlfinlab/issues/448>`_ might help.
