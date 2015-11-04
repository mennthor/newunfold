#-*-coding:utf8-*-

import numpy as np
import matplotlib.pyplot as plt
import mc_data_gen
import unfold


N = 100000
xl = 0
xh = 2

mcd = mc_data_gen.FlatUnfoldData(N=N, range=[xl, xh])
true, meas = mcd.get_mc_sample()

unf = unfold.unfold(
	range_true=[xl, xh],
	range_meas=[xl, xh],
	nbins_true=20,
	nbins_meas=20,
	)

A = unf.fit(true["data"], meas["data"], meas["weight"])

# Unfold the true distribtuion used for training to test the unfolding
predicted_res = unf.predict(true["data"])

# Simpe inversion for comparison
predicted_simple_inv = unf.predict_by_inverse(true["data"])

# Bin true MC for comparison
hist, bins = np.histogram(true["data"], bins=20, range=[xl, xh], density=False)
# print("True :")
# print(hist)

# print("")
# print("Predicted :")
# print(predicted_res.x)

# print("")
# print("Simple inv pred :")
# print(predicted_simple_inv)

# Plot
fig, ax = plt.subplots(1, 1)

binmids = 0.5 * (bins[:-1] + bins[1:])
ax.plot(binmids, hist, drawstyle="steps-mid")
ax.plot(binmids, predicted_res.x, drawstyle="steps-mid")
ax.plot(binmids, predicted_simple_inv, drawstyle="steps-mid")

ax.set_yscale("log", nonposy="clip")
ax.set_ylim((1, 1e7))

ax.set_xlabel("bin number")
ax.set_ylabel("entries")
plt.show()
