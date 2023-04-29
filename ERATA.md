# Erata

## Paper
* Logistic (Verhulst) differential equation (table 3), N -> K, source:
    Bacaër, N. (2011). Verhulst and the logistic equation (1838).*IA short history of mathematical population dynamics* (Vol. 618). London: Springer.

* Figure 1C: x-as is displayed as Time(days) but it actually needs to be Time(Weeks) since the authors convert their data to the amount of weeks.

* By comparing Table 1 and Fig 1D we can see that the author messed up the labeling of the studies. This creates an unreadable effect and makes it hard to understand which data belongs to which study. 

## Code
https://github.com/KatherLab/ImmunotherapyModels/blob/main/FitFunctions.py
* `FitFunctions.py:36` `dim**2/3` -> `dim**(2/3)`

## Source
* The authors don't provide a mapping to the named studies in the paper to their anonymized studies.

Bacaër, N. (2011). A short history of mathematical population dynamics (Vol. 618). London: Springer.