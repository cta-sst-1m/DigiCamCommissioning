.. _analysis.analyse_dark_step_function:

==============================================
Parameters extraction from dark runs
==============================================

.. currentmodule:: analysis.analyse_dark_step_function

The aim of this module is to evaluate the parameters of the pixels
from dark runs. There are several analysis implemented:

  * Step function

  * Dark ADC

  * SPE


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Expected input data
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Any runs without light. Baseline is evaluated event by event,
which implies the N first samples are not to be used in analysis


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Output data
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 1: Histogram building
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 2: Analysis
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 3: Display
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
