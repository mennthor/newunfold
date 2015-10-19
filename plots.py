#-*-coding:utf8-*-

"""
Create (normed) hists with correctly scaled errorbars.
"""

import numpy as np
import matplotlib.pyplot as plt
import mc_data_gen

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
	weights = kwargs.pop("weights", np.ones_like(data))

	# Create hist
	hist, bins = np.histogram(data, bins=nbins, range=range,
		normed=False, weights=weights)
	# Total num of events with weighting
	nevents = np.sum(hist)
	# area under hist = sum_i (binwidth_i * nevents_i
	#                 = intervall / nbins * nevents (for equal bin widths)
	if normed == True:
		hnorm = nbins / float(nevents * (xh - xl))
	else:
		hnorm = 1.
	# First calculate poissonian errors, than scale hist otself
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
	ax.hist(data, bins=nbins, normed=normed, range=range,
		weights=weights, color=color, **kwargs)
	ax.errorbar(binmids, hist, yerr=yerr, fmt=",", color=color)


N = 100000
xl = 0
xh = 2

# Plot each training and test data
for typ in ["train", "test"]:
	# Get correct MC data
	if typ == "train":
		mcd = mc_data_gen.FlatUnfoldData(N=N, range=[xl, xh])
	else:
		mcd = mc_data_gen.LorentzianUnfoldData(N=N, range=[xl, xh])

	true, meas = mcd.get_mc_sample()

	fig, ax = plt.subplots(1, 1)
	nbins = 80
	normed = True
	histerr(ax, true["data"], normed=normed, nbins=nbins, range=[xl, xh],
		weights=true["weight"],
		histtype="step", color=color["r"], label=r"true")
	histerr(ax, meas["data"], normed=normed, nbins=nbins, range=[xl, xh],
		weights=meas["weight"],
		histtype="step", color=color["b"], label=r"measured")

	# Plot true pdf
	x = np.linspace(xl, xh, 1000)
	y = mcd._pdf(x, xl, xh)
	ax.plot(x, y, label=r"true pdf", color=color["k"])

	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"probability")
	ax.set_title("$N = {{{}}}$ events before detector".format(N))
	ax.legend(loc="best")
	fig.tight_layout()
	fig.savefig("{}.png".format(typ), dpi=300, bbox_inches="tight")






