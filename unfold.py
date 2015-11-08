#-*-coding:utf8-*-

import numpy as np
import scipy.optimize as sco

class unfold():
	"""
	Unfolding class. Structure oriented on scikit learn.

	_Should later be defined as a base class. Implement unfold simple, advanced,
	etc. in subclasses._
	_Options for regularization: Cutoff or Thikhonov_

	The object gets the settings, like degrees of freedom, which define
	the unfolding model.

	The method fit trains the model on training data, eg. MC data.

	The method predict applies the trained model to test or real data.

	Note:
	The number of bins used to bin the true MC distribution is the same as the
	number of bins used for the binning of the unfolded data. This is because
	the number of rows of the matrix and the vector must match in the
	unfolding equation.
	"""
	def __init__(self, **kwargs):
		"""
		Define model parameters.

		Uses equal sized binning only. Specify number of bins and range.

		Parameters
		----------
		nbins_* : int
			*true : Number of bins for the true MC trainig data.
			*meas : Number of bins for the measured MC training data.
			*pred : Number of bins for the predicted unfolded data.
			Default values are 10.
		range_* : ndarray
			*true : Range for the binning of the true MC training data.
			*meas : Range for the binning of the measured MC training data.
			*pred : Range for the binning of the predicted unfolded data.
			Default ranges are [0, 1].
		"""
		# self.ndof = kwargs.pop("ndof", 10)

		self.range_true = kwargs.pop("range_true", [0, 1])
		self.range_meas = kwargs.pop("range_meas", [0, 1])
		self.range_pred = kwargs.pop("range_pred", [0, 1])

		self.nbins_true = kwargs.pop("nbins_true", 10)
		self.nbins_meas = kwargs.pop("nbins_meas", 10)
		self.nbins_pred = kwargs.pop("nbins_pred", 10)


	def fit(self, true, meas, meas_weights):
		"""
		Fit model to training data from MC.

		True and measured MC data are used in pairs (true[i], meas[i])
		to map the deetector influence.

		Parameters
		----------
		true : ndarray
			One dimensional unbinned true MC data used to build the
			response matrix.
		measured : ndarray
			One dimensional unbinned measured MC data used to build the
			response matrix.
		meas_weights : ndarray
			Weights per event of the measured distribution.
		"""
		# Create equally spaced bins in given range
		bins_true = np.linspace(
			self.range_true[0], self.range_true[1], self.nbins_true + 1)
		self.h_true = bins_true[1] - bins_true[0]
		bins_meas = np.linspace(
			self.range_meas[0], self.range_meas[1], self.nbins_meas + 1)
		self.h_meas = bins_meas[1] - bins_meas[0]

		# Get bin indices for each single event
		bin_index_true = np.digitize(true, bins=bins_true)
		bin_index_meas = np.digitize(meas, bins=bins_meas)

		# Build the response matrix A:
		# The entries of the response matrix A are integers.
		# Element aij of the matrix A is the number of MC events from bin j
		# of the true distribution x, which are measured in bin i of the
		# measured distributioin y.
		# So A has nbins_meas rows (index i) and nbins_true cols (index j)
		self.A = np.zeros((self.nbins_meas, self.nbins_true))
		# Loop over bin indices and count events
		for j in np.arange(0, self.nbins_true):
			for i in np.arange(0, self.nbins_meas):
				# np.digitize starts with 1 as first bin, so correct for that
				# by raising the bin indices
				mask = np.logical_and(
					bin_index_true == j+1,
					bin_index_meas == i+1,
					)
				# Nor every column by bin entry true_j. TODO: Use true weights
				# Without acceptance it would be: sum_i aij/norm_j = 1
				norm_j = np.sum(bin_index_true == j+1)
				# Sum over measured event weights to take the acceptance
				# into account
				self.A[i][j] = np.sum(meas_weights[mask]) / norm_j

		return self.A


	def predict(self, data, data_weights=None, ndof=10):
		"""
		Use trained model matrix A on prediction data.

		Parameters
		----------
		ndof : float
			Degress of freedom for the regularization. Default: 10.

		Returns
		-------
		pred : ndarray
			Array of predicted, unfolded distribution, calculated by applying
			the trained response matrix to the real measured data.
		"""
		def reg(x):
			"""
			Return the regularization term for a given set of bin entires x
			describing the target function.

			Calculate regularization term dependent on the curvature of
			histogram described by the current x values, which is the
			discretized integral over second derivation squared:
				   int (f''(x))^2 dx
				-> sum (f''(x))^2 * h
			with the second numerical derivate at every position x[j]
				f'' = (f(x[j-1]) - 2f(x[j]) + f(x[j+1])) / h^2
			and the bin width (here equal for every bin)
				h = x[j] - x[j-1] = x[1] - x[0] = const
			"""
			f2_squared = 0.
			for j in np.arange(1, self.nbins_true-1):
				f2_squared += (x[j-1] - 2*x[j] + x[j+1]) / self.h_true**3

			regularization = 0.5 * ndof * f2_squared

			return regularization

		def chi2(x, *args):
			"""
			Approximating chi-square fit to obtain initial values for the
			likelihood fit.

			Returns
			-------
			chi2 : float
				Value of the minimizer function for a specific choice of x
			"""
			y_meas = np.array(args)
			# Expected number of events for a specific choice of x
			y = np.dot(self.A, x)
			chi2 = np.sum( (y_meas - y)**2 / y_meas ) + reg(x)

			return chi2

		def llh(x, *args):
			"""
			Negative log-likelihood function used to fit the components
			of vector x, describing the bin contents of the unfolded
			distribution.

			Parameters
			----------
			x : ndarray
				Function values which ought to be minimized.
			*args : tuple
				Additional paramters:
					arg[0] : ndarray of histogrammed measured data

			Returns
			-------
			llh : float
				Value of the minimizer function for a specific choice of x
			"""
			y_meas = np.array(args)
			# Expected number of events for a specific choice of x
			y = np.dot(self.A, x)
			llh = np.sum( y - y_meas * np.log(y)) + reg(x)

			return llh

		# First, bin the measured data. No need to track anything, just
		# make a simple histogram with nbins_meas as in the fit method.
		# No need for the bin edges, so discard.
		meas, _ = np.histogram(
			data,
			bins=self.nbins_meas,
			range=[self.range_meas[0], self.range_meas[1]],
			density=False,
			weights=data_weights,
			)

		# Minimizer setup
		# The measured binned data is needed in the fit functions
		args = tuple(meas)
		method = "L-BFGS-B"
		# Hist entries need to be positive
		bounds = tuple((0, None) for i in range(self.nbins_true))

		# Init preliminary fir with the measured bin entries
		x0 = meas
		x0 = np.ones_like(meas)
		# get preliminary fit to use as initial values for the likelihood fit
		preliminary_res = sco.minimize(
			chi2,
			x0,
			args=args,
			bounds=bounds,
			method=method,
			jac=False
			)

		# Now do the llh fit with the previous fit result as initial values
		x0 = preliminary_res.x
		print("Preliminary :  {}".format(x0))
		predicted_res = sco.minimize(
			llh,
			x0,
			args=args,
			bounds=bounds,
			method=method,
			jac=False
			)

		return predicted_res


	def predict_by_inverse(self, data, data_weights=None):
		"""
		Unfold by simple matrix inversion
		"""
		s = np.shape(self.A)
		if(not len(s) == 2):
			if(not s[0] == s[1]):
				print("Error: Simple inversion only works with a square matrix.")
				return

		meas, _ = np.histogram(
		data,
		bins=self.nbins_meas,
		range=[self.range_meas[0], self.range_meas[1]],
		density=False,
		weights=data_weights,
		)

		predicted = np.dot(np.linalg.inv(self.A), meas)

		return predicted






