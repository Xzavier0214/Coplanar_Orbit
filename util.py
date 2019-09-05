from numpy import sqrt

RE = 6378.137e3
MU = 3.986004418e14

DU = RE
TU = sqrt(DU**3 / MU)
VU = DU / TU
GU = DU / TU**2
