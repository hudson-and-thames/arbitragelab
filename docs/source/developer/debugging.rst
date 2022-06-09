.. _developer-debugging:

===========================
Documentation for Debugging
===========================

This section of the documentation contains tools to simplify the process of debugging programs
that use the ArbitrageLab package.

Raw Documentation
#################

Using this link, one can download a PDF file containing raw version of the documentation, which
includes descriptions of all public and private functions, classes, and methods available in each
module of the ArbitrageLab package.

.. raw:: html

    <style>
      .special {
        display: inline-block;
        background-color: #0399AB;
        color: #eeeeee;
        text-align: center;
        font-size: 180%;
        padding: 15px;
        width: 100%;
        transition: all 0.5s;
        cursor: pointer;
        font-family: 'Josefin Sans';
      }
      .special span {
        cursor: pointer;
        display: inline-block;
        position: relative;
        transition: 0.5s;
      }
      .special span:after {
        content: '\00bb';
        position: absolute;
        opacity: 0;
        top: 0;
        right: -20px;
        transition: 0.5s;
      }
      .special:hover {
        background-color: #e7f2fa;
        color: #000000;
      }
      .special:hover span {
        padding-right: 25px;
      }
      .special:hover span:after {
        opacity: 1;
        right: 0;
      }
    </style>

    <button class="special" onclick="window.open('https://hudsonthames.org/other/arbitragelab_raw_docs.pdf','_blank')">
      <span>Download Raw Documentation</span>
    </button>

|

----

UML Diagram
###########

The link below leads to a UML diagram of the ArbitrageLab package classes in .dot format.
To use the .dot file, one may want to download the `Graphviz <https://graphviz.org/>`__ open-source software.

.. raw:: html

    <style>
      .special {
        display: inline-block;
        background-color: #0399AB;
        color: #eeeeee;
        text-align: center;
        font-size: 180%;
        padding: 15px;
        width: 100%;
        transition: all 0.5s;
        cursor: pointer;
        font-family: 'Josefin Sans';
      }
      .special span {
        cursor: pointer;
        display: inline-block;
        position: relative;
        transition: 0.5s;
      }
      .special span:after {
        content: '\00bb';
        position: absolute;
        opacity: 0;
        top: 0;
        right: -20px;
        transition: 0.5s;
      }
      .special:hover {
        background-color: #e7f2fa;
        color: #000000;
      }
      .special:hover span {
        padding-right: 25px;
      }
      .special:hover span:after {
        opacity: 1;
        right: 0;
      }
    </style>

    <button class="special" onclick="window.open('https://hudsonthames.org/other/arbitragelab_uml.zip','_blank')">
      <span>Download UML Diagram</span>
    </button>
