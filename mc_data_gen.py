#-*-coding:utf8-*-

"""
Create true and measured distributions for training and
testing of the unfolding algorithm.
"""

from __future__ import print_function, division
import numpy as np
import scipy.integrate as scig
import scipy.optimize as sco

class NullUnfoldData(object):
	"""
	Base class for generating training and test data from
	different distributions.

	All events have a unique ID in range [0, N-1] and get the weight 1.

	Events that are rejected in the acceptance step have weight 0.

	Parameters
	----------
	range : tuple of floats
		Intervall [range[0], range[1]] in which the true pdf is defined.
		It should be 0<=range[0]<range[1].
		If range[0] is <0, the shift function shifts every number<0 to 0.
	N : int
		Number of points sampled from the pdf for the true distribution

	Functions
	---------
	get_mc_sample
		Returns the internally stored MC training and test data.

	Attributes
	----------
	true : recarray
		True MC data.
	meas : recarray
		Measured MC data.
	"""
	def __init__(self, **kwargs):
		self.range = kwargs.pop("range", (0, 2))
		self.N = kwargs.pop("N", 10000)
		self.xl = self.range[0]
		self.xh = self.range[1]


	def get_mc_sample(self):
		"""
		Calls the internal MC generation functions and returns the stored
		data as recarrays.

		Returns
		-------
		true : recarray
			True MC data. Fields: (ID, data, weight)
		meas : recarray
			Measured MC data. Fields: (ID, data, weight)
		"""
		self._gen_true_sample()
		self._gen_meas_sample()
		return self.true, self.meas


	def __raise__(self, *args, **kwargs):
		raise NotImplementedError(
			"NullUnfoldData only to be used as abstract superclass".format(
				self.__repr__()))


	def _pdf(self, x, xl, xh):
		"""
		The generating pdf of the true function for a specific distribution.
		"""
		self.__raise__()


	def _gen_true_sample(self):
		"""
		Return the specific true MC data for a specific distribution.
		"""
		self.__raise__()


	def _gen_meas_sample(self):
		"""
		Use the generated true sample to apply acceptance correction, shifting
		and gaussian smearing to obtain a measured sample.
		Because of the limited acceptance, the sample may (very likely) contain
		less than N numbers.

		This simulates the detector and is decoupled from the
		generating true distribution.

		Stores generated events in a np.recarray with fields
			(ID, data, weight)
		"""
		# Test if true sample is already generated
		if self.true is None:
			self._gen_true_sample()

		# Set up weights, initially al events have the same weight
		weights = np.ones(self.N)
		data = self.true["data"]

		# Acceptance: Loose (reject) event if (rnd > acceptance function)
		yl = 0.5
		ym = 1
		yh = 0.5
		acceptance = ( np.random.uniform(size=self.N) <=
			self._parabola(data, yl, ym, yh) )
		# If not accepted, set weight to 0
		weights[~acceptance] = 0.

		# Shift values systematically
		# y_shift = x - 0.2 * x**2 / 4.
		xm = 0.5 * (self.xl + self.xh)
		# negativ values make no sense
		yl = np.maximum(self.xl, 0.)
		ym = self.xl + 0.95 * (xm - self.xl)
		yh = self.xl + 0.9 * (self.xh - self.xl)
		measured = self._parabola(data, yl, ym, yh)
		# Correct numerical errors at the lower boundary. Events should stay >= xl
		measured[measured<self.xl] = self.xl

		# Smearing, add gaussian to measured values
		sigma = (self.xh - self.xl) / 20.
		# Keep track of values smeared outside the boundaries
		overflow = np.ones_like(measured, dtype=bool)
		presmeared = np.copy(measured)
		# If values are smeareded to outside the bounds try again
		while np.sum(overflow)>0:
			measured[overflow] = presmeared[overflow] + np.random.normal(
				loc=0, scale=sigma, size=np.sum(overflow))
			overflow = np.logical_or(measured<self.xl, measured>self.xh)

		# Combine to recarray with fields (ID, data, weight)
		self.meas = np.empty((self.N, ), dtype=[
			("ID", int), ("data", np.float), ("weight", np.float)])
		self.meas["ID"] = np.arange(self.N)
		self.meas["data"] = measured
		self.meas["weight"] = weights

		return


	def _parabola(self, x, yl, ym, yh):
		"""
		Parabola through the three points
			(xl | yl), (xm | ym), (xh | yh)
		with xm = 0.5 * (xl + xh).

		Solution is analytically computed.

		A parabola is used for both shifting the data and for the acceptance
		probability.

		Parameters
		----------
		x : ndarray
			x-values where the function is evaluated
		y* : float
			y-values of the three points defining the parabola

		Returns
		-------
		f : float
			Function values at x
		"""
		xm = 0.5 * (self.xl + self.xh)

		# coefficient functions
		a = lambda xl, xm, xh, yl, ym, yh: ( ((yl-yh)*(xm-xh) - (ym-yh)*(xl-xh)) /
			((xl**2-xh**2)*(xm-xh) - (xm**2-xh**2)*(xl-xh)) )
		b = lambda a, xm, xh, rn, yh: ((ym-yh) - (xm**2-xh**2) * a) / (xm-xh)
		c = lambda a, b, xh, yh: yh - a*xh**2 - b*xh

		# Function values
		y = lambda x, a, b, c: a * x**2 + b * x + c

		a_ = a(self.xl, xm, self.xh, yl, ym, yh)
		b_ = b(a_, xm, self.xh, ym, yh)
		c_ = c(a_, b_, self.xh, yh)

		f = np.copy(y(x, a_, b_, c_))

		return f



class LorentzianUnfoldData(NullUnfoldData):
	"""
	Class for generating test data from the triple lorentzian used
	for unfolding tests by Blobel.

	This is used as a standard example for an unfolding trained with
	a flat distribution.

	Gerates a true sample with the rejection method and a measured sample
	by applying a acceptance correction, a systematic shift and a gaussian
	smearing.
	"""
	def __init__(self, **kwargs):
		super(LorentzianUnfoldData, self).__init__(**kwargs)


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


	def _gen_true_sample(self):
		"""
		Get N random numbers sampled from the true pdf using the
		rejection method.

		Stores generated events in a np.recarray with fields
			(ID, data, weight)
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
		data = []
		totgen = 0
		# Generate N samples each try and append non rejected. If total number
		# is above N, use the first N random numbers for the sample.
		while len(data)<=self.N:
			# Comparison function g(vn)=k
			vn = np.random.uniform(low=self.xl, high=self.xh, size=self.N)
			# Get pdf values from f(vn), which shall be sampled from
			fvn = self._pdf(vn, self.xl, self.xh)
			# Accept if vn * uniform[0,1] * g(vn) < f(vn)
			accept = (k  * np.random.uniform(size=self.N) < fvn)
			# Append all accepted
			data.extend(vn[accept])
			# Count total generated randonm numbers for performance information
			totgen += self.N
		# Save only the requested N random numbers
		self.data = np.array(data[:self.N])

		# Combine to recarray with fields (ID, data, weight)
		self.true = np.empty((self.N, ), dtype=[
			("ID", int), ("data", np.float), ("weight", np.float)])
		self.true["ID"] = np.arange(self.N)
		self.true["data"] = self.data
		self.true["weight"] = np.ones_like(self.data)

		return



class FlatUnfoldData(NullUnfoldData):
	"""
	Flat true distribution, used for training the unfolding model.
	"""
	def __init__(self, **kwargs):
		super(FlatUnfoldData, self).__init__(**kwargs)


	def _pdf(self, x, xl, xh):
		"""
		Simple flat distribution in intervall [xl, xh].
		"""
		f = np.ones_like(x) / float(self.xh - self.xl)

		return f


	def _gen_true_sample(self):
		"""
		Generate flat distribution with N entries.

		Stores generated events in a np.recarray with fields
			(ID, data, weight)
		"""
		# Generate N unfiormly distributed events in [xl, xh]
		self.data = np.random.uniform(low=self.xl, high=self.xh, size=self.N)

		# Combine to recarray with fields (ID, data, weight)
		self.true = np.empty((self.N, ), dtype=[
			("ID", int), ("data", np.float), ("weight", np.float)])
		self.true["ID"] = np.arange(self.N)
		self.true["data"] = self.data
		self.true["weight"] = np.ones_like(self.data)

		return




