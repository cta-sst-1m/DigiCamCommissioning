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

    * Step function : dark count rate, crosstalk

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 1: Histogram building
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  * Step function
The histogram is build using the build argument : h_type='STEPFUNCTION'.
The histogram represents the number of detected peaks above a certain threshold.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 2: Analysis
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    * Step function
The analysis of the step function :math:`F(T)` computes the number of detected peaks for the 0.5 p.e.
and the 1.5 p.e. The dark count rate :math:`f_{dark}` and crosstalk :math:`XT` are :

.. math::
    f_{dark} = \frac{F(T=0.5 \text{ p.e.})}{T_{window}}
.. math::
    XT = \frac{F(T=1.5 \text{ p.e.})}{F(T=0.5 \text{ p.e.})}

where :math:`T_{window}` is the total time



++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 3: Display
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
