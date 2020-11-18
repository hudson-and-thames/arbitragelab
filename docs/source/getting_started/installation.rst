
============
Installation
============

Recommended Versions
####################

* Anaconda
* Python 3.7 and up.

Installation
############

Ubuntu Linux
************

0. Set up Git (if you haven't already, the following `link <https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git>`__ provides a nice guide.)
1. Make sure you install the latest version of the Anaconda distribution. To do this you can follow the install and update instructions found on this `link <https://www.anaconda.com/products/individual>`_
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
      * To confirm your new env variable is active: ``echo $ARBLAB_API_KEY``

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

      pip install git+https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@github.com/hudson-and-thames/arbitragelab.git@master

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

7. Install arbitragelab into your python environment via the terminal.

   Please make sure to use this exact statement:

   .. code-block::

      pip install git+https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@github.com/hudson-and-thames/arbitragelab.git@master

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

   conda create -n <env name> python=3.7 anaconda

4. Activate the new environment:

.. code-block::

   conda activate <env name>

5. Purchase ArbitrageLab from the `Hudson & Thames website <https://app.hudsonthames.org/auth/signin>`__. This will provide you with an API key.

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

      pip install git+https://1fed2947109cfffdd6aaf615ea84a82be897c4b9@github.com/hudson-and-thames/arbitragelab.git@master

.. tip::

   * We have added error handling which will raise an error if your environment variables are incorrect.
   * If you are having problems with the installation, please ping us on Slack and we will be able to assist.

Important Notes
###############

* ArbitrageLab requires an internet connection when you import the library. This checks that your API key is valid.
* We have added analytics to the library which will let us know in which city the function call was made and which functions were called, but it shares no personal data and goes via Google Analytics.
  This to help aid our team to improve the functionality that is used the most (standard practice with SaaS products).