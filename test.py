
import itertools
import numpy as np

n_colrow = 3
d_min = np.linspace(.25, .45, n_colrow)
softness = np.linspace(0, 1, n_colrow)
#
# for (i, (d_min_, softness_)) in enumerate(itertools.product(d_min, softness)):
#     print("Dmin: {d_min:.2f}, Softness: {soft:.2f}".format(
#         d_min=d_min_, soft=softness_))


print([f"Dmin: {d_min_}, Softness: {soft_}" for (i, (d_min_, soft_)) in enumerate(itertools.product(d_min, softness))])

