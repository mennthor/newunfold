#-*-coding:utf8-*-

import numpy as np
import matplotlib.pyplot as plt
import mc_data_gen
import unfold as unf

"""
Predict with multplie techniques and plot the resulting distributions.
"""

N = 10000
xl = 0
xh = 2
nbins_true=30
nbins_meas=15

mcd = mc_data_gen.FlatUnfoldData(N=N, range=[xl, xh])
true, meas = mcd.get_mc_sample()

#### Create response matrix once
unf = unf.unfold(
	range_true=[xl, xh],
	range_meas=[xl, xh],
	nbins_true=nbins_true,
	nbins_meas=nbins_meas,
	)
A = unf.fit(true["data"], meas["data"], meas["weight"])
square = (nbins_true == nbins_meas)

#### Fit true distribution using various models
# LLH fit unfolding and different regularizations
predicted_res = unf.predict(true["data"], ndof=0)
predicted_res_t1 = unf.predict(true["data"], ndof=1)
predicted_res_t10 = unf.predict(true["data"], ndof=10)
predicted_res_tinf = unf.predict(true["data"], ndof=1.e10)

# Simple inversion if square (nbins_true = nbins_meas)
if square:
	predicted_inverse = unf.predict_by_inverse(true["data"])

# Pseudoinverse with least squares. If square, should be equal to simple inversion
predicted_pseudoinverse = unf.predict_by_pseudoinverse(true["data"])

# True/meas distribution. Acceptance??
hist, bins = np.histogram(true["data"], bins=nbins_true, range=[xl, xh], density=False)
hist_meas, bins_meas = np.histogram(meas["data"], bins=nbins_meas, range=[xl, xh], density=False)

#### Plot
fig, ax = plt.subplots(1, 1, figsize=(16,9))

binmids = 0.5 * (bins[:-1] + bins[1:])
binmids_meas = 0.5 * (bins_meas[:-1] + bins_meas[1:])
ax.plot(binmids, hist, drawstyle="steps-mid", label="true")
ax.plot(binmids_meas, hist_meas, drawstyle="steps-mid", label="meas")
ax.plot(binmids, predicted_res.x, drawstyle="steps-mid", label="llh fit")
ax.plot(binmids, predicted_res_t1.x, drawstyle="steps-mid", label="llh fit, t=1")
ax.plot(binmids, predicted_res_t10.x, drawstyle="steps-mid", label="llh fit, t=10")
ax.plot(binmids, predicted_res_tinf.x, drawstyle="steps-mid", label="llh fit, t=inf")
if square:
	ax.plot(binmids, predicted_inverse, drawstyle="steps-mid", label="inverse")
ax.plot(binmids, predicted_pseudoinverse.x, drawstyle="steps-mid", label="pseudoinv")

ax.set_yscale("log", nonposy="clip")
ax.set_ylim((1e-2, 1e9))

ax.set_xlabel("bin number")
ax.set_ylabel("entries")
ax.legend(loc="best")
plt.show()
