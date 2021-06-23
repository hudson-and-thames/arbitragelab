.. _getting_started-installation:

============
Installation
============

Recommended Versions
####################

* Anaconda
* Python 3.8 and up.

Installation
############

Ubuntu Linux
************

0. Set up Git (if you haven't already, the following `link <https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git>`__ provides a nice guide.)
1. Make sure you install the latest version of the Anaconda distribution. To do this you can follow the install and update instructions found on this `link <https://www.anaconda.com/products/individual>`_
2. Launch a terminal
3. Create a New Conda Environment. From terminal.

   .. code-block::

      conda create -n <env name> python=3.8 anaconda

   Accept all the requests to install.

4. Now activate the environment with:

   .. code-block::

      source activate <env name>

5. Purchase ArbitrageLab from the `Hudson & Thames website <https://portal.hudsonthames.org>`__. This will provide you with an API key.

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
      * To confirm your new env variable is active: ``echo $ARBLAB_API_KEY``

      .. tip::

         * If you are using Ubuntu on WSL (Windows Subsystem for Linux), then you should add this environment variable
           to ~/.profile. Since you always load WSL from bash, this would make sure that the environment variable could
           be loaded each time you start the virtual machine, which in turn ensured that Python can pick it up.


   6.2 The Easy Way:

      If you don't want the key to persist on your local machine, you can always declare it each time, before you import ArbitrageLab.

      * In your python script or notebook, add the following line before you import ArbitrageLab:

      .. code::

         import os
         os.environ['ARBLAB_API_KEY'] = "426303b02cb7475984b2d484319062a0"
         import arbitragelab as al

      .. tip::

         If you are running Ubuntu on a virtual machine, you may find it easiest to use the ``os.environ`` method.

7. Install arbitragelab into your python environment via the terminal.

   Please make sure to use this exact statement:

   .. code-block::

      pip install git+https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@github.com/hudson-and-thames/arbitragelab.git@0.5.0

8. (Optional) **Only if you want to use the ML Approach Module**, install the TensorFlow and Keras packages.
   Note that you should have pip version "pip==20.1.1" to do this. Supported TensorFlow and Keras versions
   are "tensorflow==2.2.1" and "keras==2.3.1".

   To change the pip version:

   .. code-block::

      pip install --user "pip==20.1.1"

   To install TensorFlow and Keras:

   .. code-block::

      pip install "tensorflow==2.2.1"
      pip install "keras==2.3.1"

   .. warning::

      You may be encountering the following errors during the installation:

      ``ERROR: tensorflow 2.2.1 has requirement h5py<2.11.0,>=2.10.0,``
      ``but you'll have h5py 3.1.0 which is incompatible.``

      ``ERROR: tensorflow 2.2.1 has requirement numpy<1.19.0,>=1.16.0,``
      ``but you'll have numpy 1.20.1 which is incompatible.``

      You can ignore these messages. They appear due to the updated dependency versions in the ArbitrageLab package.

      All the ArbitrageLab functionality still works as expected.

.. tip::

   * We have added error handling which will raise an error if your environment variables are incorrect.
   * If you are having problems with the installation, please ping us on Slack and we will be able to assist.


Mac OS X
********

0. Set up Git (if you haven't already, the following `link <https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git>`__ provides a nice guide.)
1. Make sure you install the latest version of the Anaconda distribution. To do this you can follow the install and update instructions found on this `link <https://www.anaconda.com/products/individual>`_
2. Launch a terminal
3. Create a New Conda Environment. From terminal.

   .. code-block::

      conda create -n <env name> python=3.8 anaconda

   Accept all the requests to install.

4. Now activate the environment with:

   .. code-block::

      source activate <env name>

5. Purchase ArbitrageLab from the `Hudson & Thames website <https://portal.hudsonthames.org>`__. This will provide you with an API key.

   .. code-block::

      Example: "26303adb02cb759b2d484233162a0"

6. Add API key as an environment variable:

   6.1 The Best Way:

      By adding the API key as an environment variable, you won't need to constantly add the key every time you import the library.

      * Open the terminal and run: ``sudo nano ~/.bash_profile``. This will open a text editor.
      * Note: If there is no file named .bash_profile, then this above nano command will create a new file named .bash_profile.
      * Add the following environment variable to the last line of the file: ``export ARBLAB_API_KEY="26303adb02cb759b2d484233162a0"``
      * Note that you must add your own API key and not the one given in this example.
      * Press ctrl+X to exit the editor. Press ‘Y’ for saving the buffer, and you will return back to the terminal screen.
      * Restart your computer. (If you skip this step, it won't register the change). The following may work to refresh your environment: ``source ~/.bash_profile``
      * To confirm your new env variable is active: ``echo $ARBLAB_API_KEY``

   6.2 The Easy Way:

      If you don't want the key to persist on your local machine, you can always declare it each time, before you import ArbitrageLab.

      * In your python script or notebook, add the following line before you import ArbitrageLab:

      .. code::

         import os
         os.environ['ARBLAB_API_KEY'] = "426303b02cb7475984b2d484319062a0"
         import arbitragelab as al

7. Install cvxpy into your conda environment via the terminal.

   .. warning::

        Please make sure to perform this step in order for the Sparse Mean-reverting Portfolio Module to work properly.

   This is needed for the cvxpy optimizers to work properly on Windows:

   .. code-block::

      conda install -c conda-forge "cvxpy=1.1.10"

8. Install arbitragelab into your python environment via the terminal.

   Please make sure to use this exact statement:

   .. code-block::

      pip install git+https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@github.com/hudson-and-thames/arbitragelab.git@0.5.0

9. (Optional) **Only if you want to use the ML Approach Module**, install the TensorFlow and Keras packages.
   Note that you should have pip version "pip==20.1.1" to do this. Supported TensorFlow and Keras versions
   are "tensorflow==2.2.1" and "keras==2.3.1".

   To change the pip version:

   .. code-block::

      pip install --user "pip==20.1.1"

   To install TensorFlow and Keras:

   .. code-block::

      pip install "tensorflow==2.2.1"
      pip install "keras==2.3.1"

   .. warning::

      You may be encountering the following errors during the installation:

      ``ERROR: tensorflow 2.2.1 has requirement h5py<2.11.0,>=2.10.0,``
      ``but you'll have h5py 3.1.0 which is incompatible.``

      ``ERROR: tensorflow 2.2.1 has requirement numpy<1.19.0,>=1.16.0,``
      ``but you'll have numpy 1.20.1 which is incompatible.``

      You can ignore these messages. They appear due to the updated dependency versions in the ArbitrageLab package.

      All the ArbitrageLab functionality still works as expected.

.. tip::

   * We have added error handling which will raise an error if your environment variables are incorrect.
   * If you are having problems with the installation, please ping us on Slack and we will be able to assist.


Windows
*******

0. Set up Git (if you haven't already, the following `link <https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git>`__ provides a nice guide.)
1. Download and install the latest version of `Anaconda 3 <https://www.anaconda.com/products/individual>`__
2. Launch Anacoda Prompt
3. Create new environment (replace <env name> with a name, for example ``arbitragelab``):

   .. code-block::

      conda create -n <env name> python=3.8 anaconda

4. Activate the new environment:

   .. code-block::

      conda activate <env name>

5. Purchase ArbitrageLab from the `Hudson & Thames website <https://portal.hudsonthames.org>`__. This will provide you with an API key.

   .. code-block::

      Example: "26303adb02cb759b2d484233162a0"

6. Add API key as an environment variable:

   6.1 The Best Way:

      By adding the API key as an environment variable, you won't need to constantly add the key every time you import the library.

      * Open command prompt as an administrator.
      * Create the variable: ``setx ARBLAB_API_KEY  "26303adb02cb759b2d484233162a0"``
      * Note that you must add your own API key and not the one given in this example.
      * Close and open a new command prompt
      * Validate that your variable has been added: ``echo %ARBLAB_API_KEY%``

   6.2 The Easy Way:

      If you don't want the key to persist on your local machine, you can always declare it each time, before you import ArbitrageLab.

      * In your python script or notebook, add the following line before you import ArbitrageLab:

      .. code::

         import os
         os.environ['ARBLAB_API_KEY'] = "426303b02cb7475984b2d484319062a0"
         import arbitragelab as al

7. Install arbitragelab into your python environment via the terminal.

   Please make sure to use this exact statement:

   .. code-block::

      pip install git+https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@github.com/hudson-and-thames/arbitragelab.git@0.5.0

8. (Optional) **Only if you want to use the ML Approach Module**, install the TensorFlow and Keras packages.
   Note that you should have pip version "pip==20.1.1" to do this. Supported TensorFlow and Keras versions
   are "tensorflow==2.2.1" and "keras==2.3.1".

   To change the pip version:

   .. code-block::

      pip install --user "pip==20.1.1"

   To install TensorFlow and Keras:

   .. code-block::

      pip install "tensorflow==2.2.1"
      pip install "keras==2.3.1"

   .. warning::

      You may be encountering the following errors during the installation:

      ``ERROR: tensorflow 2.2.1 has requirement h5py<2.11.0,>=2.10.0,``
      ``but you'll have h5py 3.1.0 which is incompatible.``

      ``ERROR: tensorflow 2.2.1 has requirement numpy<1.19.0,>=1.16.0,``
      ``but you'll have numpy 1.20.1 which is incompatible.``

      You can ignore these messages. They appear due to the updated dependency versions in the ArbitrageLab package.

      All the ArbitrageLab functionality still works as expected.

.. tip::

   * We have added error handling which will raise an error if your environment variables are incorrect.
   * If you are having problems with the installation, please ping us on Slack and we will be able to assist.

Important Notes
###############
* You may need to `install Cython <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`__ if your distribution hasn't already.
* ArbitrageLab requires an internet connection when you import the library. This checks that your API key is valid.
* We have added analytics to the library which will let us know in which city the function call was made and which functions were called, but it shares no personal data and goes via Google Analytics.
  This to help aid our team to improve the functionality that is used the most (standard practice with SaaS products).