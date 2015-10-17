#-*-coding:utf8-*-

"""
Create and plot the Blobel triple lorentzian testing pdf.
"""

from __future__ import print_function, division
import numpy as np
import scipy.integrate as scig
import scipy.optimize as sco

class BlobelTestLorentzian():
	"""
	Class for generating test data from the triple lorentzian used
	for unfolding tests by Blobel.

	Gerates a true sample with the rejection method and a measured sample
	by applying a acceptance correction, a systematic shift and a gaussian
	smearing.

	Parameters
	----------
	range : tuple of floats
		Intervall [range[0], range[1]] in which the true pdf is defined
	N : int
		Number of points sampled from the pdf for the true distribution

	"""
	def __init__(self, **kwargs):
		self.range = kwargs.pop("range", (0, 2))
		self.xl = self.range[0]
		self.xh = self.range[1]
		self.N = kwargs.pop(N, 10000)


	def _pdf(self, x, xl, xh):
		"""
		Sum of three lorentz functions, normed to unity in intervall [xl, xh].
		The normalization is calculated analytically.
		"""
		bk = [1, 10, 5]
		xk = [0.4, 0.8, 1.5]
		gk = [2, 0.2, 0.2]

		f = 0
		norm = 0
		for bi, xi, gi in zip(bk, xk, gk):
			f += bi * gi**2 / ((x - xi)**2 + gi**2)
			norm += ( bi * gi * np.arctan((self.xh - xi) / gi)
				- bi * gi * np.arctan((self.xl - xi) / gi) )

		return f / norm

	# Sample with rejection method
	def get_true_sample(self):
		"""
		Get N random numbers sampled from the true pdf using the
		rejection method.

		Returns
		-------
		true : ndarray
			True random sample with size N.
		"""
		def minpdf(x, xl, xh):
			"""
			Function wrapper for the minimizer.

			As there are only minimizers in scipy, invert the sign, which
			makes the maximum the minimum.

			Arguments are the x values as an array and the lower/upper boundary.
			"""
			return -self._pdf(x, xl, xh)
		# First get the global maximum of the pdf used for rejection sampling.
		# Use x0=0 to be sure to find the correct left one.
		optres = sco.minimize(
			minpdf,
			x0=[2,],
			args=(self.xl, self.xh),
			method='L-BFGS-B'
			)
		# Get minimized values, don't forget to switch the sign back
		xmax = optres.x[0]
		ymax = -optres.fun[0]
		# Sample N points with rejection. Use a uniform distribution with a
		# scaling factor k for comparison. Set k slightly above the found
		# maximum to be sure to cover everything.
		k = 1.01 * ymax
		true = []
		totgen = 0
		# Generate N samples each try and append non rejected. If total number
		# is above N, use the first N random numbers for the sample.
		while len(true)<=N:
			# Comparison function g(vn)=k
			vn = np.random.uniform(low=xl, high=xh, size=N)
			# Get pdf values from f(vn), which shall be sampled from
			fvn = pdf(vn, xl, xh)
			# Accept if vn * uniform[0,1] * g(vn) < f(vn)
			accept = (k  * np.random.uniform(size=N) < fvn)
			# Append all accepted
			true.extend(vn[accept])
			# Count total generated randonm numbers for performance information
			totgen += N
		# Save only the requested N random numbers
		self.true = np.array(true[:N])

		return self.true

	def get_meas_sample(self):
		"""
		Use the generated true sample to apply acceptance correction, shifting
		and gaussian smearing to obtain a measured sample.
		Because of the limited acceptance, the sample may (very likely) contain
		less than N numbers.

		Returns
		-------
		measured : ndarray
			Measured random sample with size probably smaller than N.
		"""
		# Test if true sample is already generated
		if self.true is None:
			self.get_true_sample()

		# Now apply all three effects one after another
		# Acceptance: Loose (reject) event if (rnd > acceptance function)
		acceptance = ( np.random.uniform(size=self.N)
			<= (1. - 0.5 * (self.true - 1)**2) )
		measured = self.true[acceptance]

		# Shift accepted values systematically
		measured = measured - 0.2 * measured**2 / 4.

		# Smearing, add gaussian to measured values
		overflow = np.ones_like(measured, dtype=bool)
		smear = np.copy(measured)
		# If values are smeared to outside the bounds try again
		while np.sum(overflow)>0:
			smear[overflow] = measured[overflow] + np.random.normal(
				loc=0, scale=0.1, size=len(measured[overflow]))
			overflow = np.logical_or(smear<xl, smear>xh)

		self.measured = smear

		return self.measured









