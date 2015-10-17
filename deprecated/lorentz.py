#-*-coding:utf8-*-

"""
Create and plot the Blobel triple lorentzian testing pdf.
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scig
import scipy.optimize as sco

def pdf(x, xl, xh):
	"""
	Sum of three lorentz functions, normed to unity in intervall [xl, xh].
	"""
	bk = [1, 10, 5]
	xk = [0.4, 0.8, 1.5]
	gk = [2, 0.2, 0.2]

	f = 0
	norm = 0
	for bi, xi, gi in zip(bk, xk, gk):
		f += bi * gi**2 / ((x - xi)**2 + gi**2)
		norm += ( bi * gi * np.arctan((xh - xi) / gi)
			- bi * gi * np.arctan((xl - xi) / gi) )

	return f / norm

# Intervall
xl = 0
xh = 2

# Normalization (should always be unity)
norm, normerr = scig.quad(pdf, xl, xh, args=(xl, xh))
print("Normalization :     {} ± {}".format(norm, normerr))

# Numerically get cdf at N point in [xl, xh]
xcdf = np.linspace(xl, xh, 100)
cdf = np.zeros_like(xcdf)
cdferr = np.zeros_like(xcdf)
# Integrate pdf from xl to x up to x=xh
for i, x in enumerate(xcdf):
	cdf[i], cdferr[i]  = scig.quad(pdf, xl, x, args=(xl, xh))

# Sample with rejection method
def get_mc(pdf, N):
	"""
	Get N random numbers
	"""
	# First get the main (left) maximum, use x0=0 to be sure to find the correct one
	def minpdf(x, xl, xh):
		"""
		Function wrapper for the minimizer.
		"""
		return -pdf(x, xl, xh)
	optres = sco.minimize(minpdf, x0=[2,], args=(xl, xh), method='L-BFGS-B')
	# Get minimized values, don't forget tp switch the sign back
	xmax = optres.x[0]
	ymax = -optres.fun[0]
	print("Maximum of pdf is at ({:5.3f} | {:5.3f})".format(xmax, ymax))
	# Sample N points with rejection. Use unfiform distribution and a scaling k
	# slightly above the found maximum to be sure to cover everything
	k = 1.01 * ymax
	print("Use rejection sampling scaling factor :  {}".format(k))
	true = []
	totgen = 0
	# Generate N samples each try and append non rejected. If total number is above
	# N, use the first N random numbers for the sample
	while len(true)<=N:
		# Comparison function g(vn)=k
		vn = np.random.uniform(low=xl, high=xh, size=N)
		# Get pdf values from wanted f(vn)
		fvn = pdf(vn, xl, xh)
		# Accept if vn * uniform[0,1] * g(vn) < f(vn)
		accept = (k  * np.random.uniform(size=N) < fvn)
		# Append all accepted
		true.extend(vn[accept])
		# Count total generated randonm numbers for performance information
		totgen += N
	# Use only the requested N random numbers
	print("    Generation efficency :  {}".format(len(true)/totgen))
	true = np.array(true[:N])
	print("    Generated {} numbers".format(len(true)))

	# Acceptance: Loose (reject) event if (rnd > acceptance function)
	acceptance = ( np.random.uniform(size=N) <= (1. - 0.5 * (true - 1)**2) )
	measured = true[acceptance]

	# Shift accepted values systematically
	measured = measured - 0.2 * measured**2 / 4.

	# Smearing, add gaussian to measured values
	overflow = np.ones_like(measured, dtype=bool)
	smear = np.copy(measured)
	# if values are smeared outside bounds try again
	while np.sum(overflow)>0:
		smear[overflow] = measured[overflow] + np.random.normal(
			loc=0, scale=0.1, size=len(measured[overflow]))
		overflow = np.logical_or(smear<xl, smear>xh)

	measured = smear

	return true, measured

N = 1000000
true, measured = get_mc(pdf, N)


# Plots
fig, ax = plt.subplots(1,1)
x = np.linspace(xl, xh, 1000)
ax.plot(x, pdf(x, xl, xh), "k-", label="pdf")
# ax.errorbar(xcdf, cdf, yerr=cdferr, fmt="ko", markersize=1, label="cdf")
# ax.axhline(1, xl, xh, ls=":", color="b", label="y=1")
# ax.axhline(k, xl, xh, ls=":", color="m", label="k={:4.2f}".format(k))
# ax.axvline(xmax, 0, 1, ls="--", color="r", label="xmax")

# Normed true hist with poissonian errorbars
nbins = 80
nevents = len(true)
hist, bins = np.histogram(true, bins=nbins, normed=False)
# area under hist = sum_i (binwidth_i * nevents_i
#                 = intervall / nbins * nevents (for equal bin widths)
hnorm = nbins / (nevents * (xh - xl))
yerr = np.sqrt(hist) * hnorm
binmids = 0.5 * (bins[:-1] + bins[1:])
ax.hist(true, bins=nbins, normed=True, histtype="step", color="b", label="True")
ax.errorbar(binmids, hist*hnorm, yerr=yerr, fmt=",", color="b")

# Normed measured hist
ax.hist(measured, bins=nbins, normed=True, histtype="step", color="r", label="measured")

ax.legend(loc="best")
plt.show()











