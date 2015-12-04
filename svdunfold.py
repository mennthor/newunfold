#-*-coding:utf8-*-

import numpy as np
np.set_printoptions(precision=3, linewidth=200)
import matplotlib.pyplot as plt
import mc_data_gen


##### Initialization #####
# 1. Define number of bins and boundaries for the measured histogram
nbins_meas = 10
range_meas = [0, 2]
bins_meas = np.linspace(range_meas[0], range_meas[1], nbins_meas + 1)
# 2. Define number of bins and boundaries common for the true unfolded
#    histogram and the initial MC distribution
nbins_true = 10
range_true = [0, 2]
bins_true = np.linspace(range_true[0], range_true[1], nbins_true + 1)
# 3. Build the second numerical derivate matrix C. Add a small diagonal
#    component psi to make sure C can be inverted
N = nbins_true
C = np.zeros((N, N))
C[  0,   0:2] = [-1,  1]
C[N-1, N-2:N] = [ 1, -1]
for i in range(1, N-1):
	C[i, i-1:i+2] = [1, -2, 1]
psi = 1.e-3
C = C + np.diag(np.zeros(N) + psi)
# 4. Calculate the inverse C^-1 (pinv might be more stable at small psi)
# Cinv = np.linalg.inv(C)
Cinv = np.linalg.pinv(C, rcond=0)
# 5. Generate initial MC histogram and the response matrix A. The entries
#    in A contain actual event numbers
Nevents = 5000
mcd = mc_data_gen.LorentzianUnfoldData(
	N=Nevents,
	range=[range_meas[0], range_meas[1]]
	)
true, meas = mcd.get_mc_sample()
true_data = true["data"]
meas_data = meas["data"]
meas_weights = meas["weight"]
bin_index_true = np.digitize(true_data, bins=bins_true)
bin_index_meas = np.digitize(meas_data, bins=bins_meas)
A = np.zeros((nbins_meas, nbins_true))
for j in np.arange(0, nbins_true):
	for i in np.arange(0, nbins_meas):
		mask = np.logical_and(
			bin_index_true == j+1,
			bin_index_meas == i+1,
			)
		A[i][j] = np.sum(meas_weights[mask])
# 6. Create the measured histogram b and its covariance matrix B. The covariance
#    is assumed diagonal with sqrt(bin_content) as a variance for each bin
b, _ = np.histogram(
			meas_data,
			bins=nbins_meas,
			range=[range_meas[0], range_meas[1]],
			density=False,
			weights=meas_weights,
			)
B = np.diag(b)

# ##### Rescaling and rotation #####
# # 1. Perform SVD of the covariance matrix B. Note: B is pos-semidef, so
# #    the pseudo inverse yields two identical transformation matrices
# Q, R, Qt = np.linalg.svd(B)
# R = np.diag(R)
# # 2. Rotate and rescale A and the measured bin entries (r.h.s. of the LGS)
# #    After that the covariance matrix is the unit matrix
# sqrtRinv = np.sqrt(np.linalg.inv(R))
# bt = np.dot(sqrtRinv, np.dot(Q, b))
# At = np.dot(sqrtRinv, np.dot(Q, A))
# # 3. Calculate the inverse of the covariance matrix X of the unfolded
# #    distribution
# # Xinv = np.zeros((nbins_true, nbins_true))
# # for j in range(nbins_true):
# # 	for k in range(nbins_true):
# # 		Xinv[j, k] = np.sum(At[:, j], At[:, k]) / (true_data[j] * true_data[k])
# # 4. Multiply At and Cinv and perform a SVD of the product
# U, S, Vt = np.linalg.svd(np.dot(At, Cinv))
# # 5. Calculate the rotated measured histogram d (r.h.s. of the rotated system)
# d = np.dot(U.T, bt)


# ##### Unfolding #####
# # 1. Plot log10|d_i| vs index i and determine the effective rank of the system
# #    from the resulting plot
# fig, ax = plt.subplots(1, 1)
# ax.hist(range(len(d)), bins=len(d), weights=np.abs(d), color="k", histtype="step")
# ax.axhline(y=1, xmin=0, xmax=1, color="r", ls="--")
# ax.set_xlabel("bin index")
# ax.set_ylabel("abs(d_i)")
# ax.set_yscale("log", nonposy="clip")
# plt.tight_layout()
# plt.savefig("logd.pdf", bbox_inches="tight")


##DEBUG
print("##### Initialization #####\n" + 60*"-")
print("1. Number of bins and boundaries for the measured histogram b")
print("nbins_meas = {}".format(nbins_meas))
print("range_meas = {}".format(range_meas))
print("bins_meas = {}".format(bins_meas))
print("")
print("2. Define num of bins, bounds for both true/init-MC distributuons")
print("nbins_true = {}".format(nbins_true))
print("range_true = {}".format(range_true))
print("bins_true = {}".format(bins_true))
print("")
print("3. Modified second derivate matrix C")
print("psi = {}".format(psi))
print(C)
print("")
print("4. Inverse C^-1")
print(Cinv)
print("")
print("5. Generate initial MC histogram and the response matrix A.")
print("Nevents = {}".format(Nevents))
print(A)
plt.matshow(A)
plt.xlabel("meas")
plt.ylabel("true")
plt.savefig("resp.pdf")
print("")
print("6. Measured histogram b and its covariance matrix B")
print(b)
print(B)

print("")
print("##### Rescaling & Rotation #####\n" + 60*"-")

print("")
print("##### Unfolding #####\n" + 60*"-")
# print(At)
# print(bt)
# plt.matshow(At)
# plt.savefig("resp_trans.pdf")
##DEBUG_END















