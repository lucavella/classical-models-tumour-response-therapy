# Erata

## Paper
* Logistic (Verhulst) differential equation (table 3), N -> K, source:
    Bacaër, N. (2011). Verhulst and the logistic equation (1838).*IA short history of mathematical population dynamics* (Vol. 618). London: Springer.

* Gompertz differential equation solution (table 3), changed, source:
    Norton, L. (1988). A Gompertzian model of human breast cancer growth. Cancer research, 48(24_Part_1), 7067-7071.
    and https://doi.org/10.1371/journal.pcbi.1003800.

* Figure 1C: x-axis is displayed as Time(days) but it actually needs to be Time(Weeks) since the authors convert their data to the amount of weeks.

* By comparing Table 1 and Fig 1D we can see that the author messed up the labeling of the studies. This creates an unreadable effect and makes it hard to understand which data belongs to which study.

* Figure 2C: log scales, but data is normalized linearly. Should data be normalized logarithmically to fit more accurate for small values?

## Code
https://github.com/KatherLab/ImmunotherapyModels/blob/main/FitFunctions.py
* `FitFunctions.py:36` `dim**2/3` -> `dim**(2/3)`

* Wrong parameter bounds for model functions (general and normal Gompertz beta lower bound = 0), source used in paper:
    Kuang, Y., Nagy, J. D., & Eikenberry, S. E. (2018). *Introduction to mathematical oncology*. CRC Press. (table 2.1)

* When converting to weeks, the code sets negative values to 0.1
    
* If LD has no value (NOT EVALUABLE, TOO SMALL TO MEASURE), paper assumes it is too small to evaluate and thus set it to a volume of 0. We use the same assumption

## Source
* The authors don't provide a mapping to the named studies in the paper to their anonymized studies.

Bacaër, N. (2011). A short history of mathematical population dynamics (Vol. 618). London: Springer.