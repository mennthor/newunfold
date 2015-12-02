#-*-coding:utf8-*-

"""
Plot the response matrix A build from flat training data.

The plots are read as following:
  Each column represents one bin j from the true distribution.
  For each column, the different rows show how the events from true bin j are
  distributed after the measurement.

The three applied detector effects are clearly visible:
  1. The shift moves the events awawy from the matrix diagonal
  2. The smearing blurs the events around the diagonal
  3. The acceptance lowers the bin content at low and high bin indices
"""

import numpy as np
import matplotlib.pyplot as plt
import mc_data_gen
import unfold

# Pretty plots, custom colors
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color = {
         "r" : "#FF5555",
         "b" : "#5DA5DA",
         "g" : "#7BAA00",
         "k" : "#262626",
         "o" : "#FFA955",
         "m" : "#9E62C4",
         "n" : "#9D7331",
         "p" : "#F590CF",
         "c" : "#6DE4DF",
         "y" : "#FFDC55",
         }


N = 100000
xl = 0
xh = 2

# mcd = mc_data_gen.LorentzianUnfoldData(N=N, range=[xl, xh])
mcd = mc_data_gen.FlatUnfoldData(N=N, range=[xl, xh])
true, meas = mcd.get_mc_sample()

unf = unfold.unfold(
	range_true=[xl, xh],
	range_meas=[xl, xh],
	nbins_true=20,
	nbins_meas=20,
	)
A = unf.fit(true["data"], meas["data"], meas["weight"])

fig, ax = plt.subplots(1, 1)

cax = ax.matshow(A, cmap="bone_r")
cbar = fig.colorbar(cax)
cbar.set_label(r"$\log_{10}(a_{ij})$")

ax.set_xlabel(r"bin $i$ of true MC distribution")
ax.set_ylabel(r"bin $j$ of measured MC distribution")

ax.set_title(r"Response matrix from MC distribution, $N={{{}}}$ events"
	.format(N))
ax.grid()
# ax.set_xlim((-5, 25))
# ax.set_ylim((25, -5))

# diagonal
ax.plot([0, 19], [0, 19], color=color["r"])

fig.tight_layout()
fig.savefig("ResponseMatrix.png", dpi=300, bbox_inches="tight")






