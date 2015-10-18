#-*-coding:utf8-*-

"""
Create (normed) hists with correctly scaled errorbars.
"""

import numpy as np
import matplotlib.pyplot as plt
import lorentz

def histerr(ax, data, nbins=20 ,**kwargs):
	"""
	Create mpl hists with poissonian errorbars, that are scaled correctly
	if the hist is normed.
	Currently works only with equally sized bins.

	Parameters
	----------
	ax : mpl axis object
		Axis on which the histogram shall be drawn.
	kwargs : dict
		Arguments passed to plt.hist function.
		"bins" is removed as only equally spaced bins are supported.
		"normed" is also used in np.histogram.
		"color" is also used plt.errorbar.

	"""
	# Remove bins from kwargs, because its not supported here
	kwargs.pop("bins", None)
	# Pop all arguments that should not be used in plt.hist only,
	# as kwargs can't be used on multiple functions when args are not supported
	normed = kwargs.pop("normed", False)
	color = kwargs.pop("color", "k")
	range = kwargs.pop("range", None)

	# Total num of events
	nevents = len(data)
	# Create hist
	hist, bins = np.histogram(data, bins=nbins, range=range, normed=False)
	# area under hist = sum_i (binwidth_i * nevents_i
	#                 = intervall / nbins * nevents (for equal bin widths)
	if normed == True:
		hnorm = nbins / float(nevents * (xh - xl))
	else:
		hnorm = 1.
	# Fisrt calculate poissonian errors, than scale hist otself
	yerr = np.sqrt(hist) * hnorm
	hist = hist * hnorm
	binmids = 0.5 * (bins[:-1] + bins[1:])
	## Draw bins as horizontal lines w/o endcaps
	# for i, _ in enumerate(hist):
	# 	ax.plot(
	# 		[bins[i], bins[i+1]],
	# 		[hist[i], hist[i]],
	# 		)
	## Draw bins as errorbars
	# ax.errorbar(binmids, hist, xerr=(bins[1]-bins[0])/2, fmt=",", color=color)
	## Draw bins with steps
	ax.hist(data, bins=nbins, normed=normed, range=range, color=color, **kwargs)
	ax.errorbar(binmids, hist, yerr=yerr, fmt=",", color=color)


N = 100000
xl = 0
xh = 2
l = lorentz.BlobelTestLorentzian(N=N, range=[xl, xh])
true, meas = l.get_mc_sample()

fig, ax = plt.subplots(1, 1)
nbins = 80
normed = True
histerr(ax, true, normed=normed, nbins=nbins, range=[xl, xh],
	histtype="step", color="r", label="true")
histerr(ax, meas, normed=normed, nbins=nbins, range=[xl, xh],
	histtype="step", color="b", label="measured")

# Plot true pdf
x = np.linspace(xl, xh, 1000)
y = l._pdf(x, xl, xh)
ax.plot(x, y, label="true pdf")

ax.legend(loc="best")
plt.show()






