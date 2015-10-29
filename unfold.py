#-*-coding:utf8-*-

import numpy as np

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
	"""
	def __init__(self, **kwargs):
		"""
		Define model parameters.

		Uses equal sized binning only. Specify number of bins and range.

		Parameters
		----------
		ndof : float
			Degress of freedom for the regularization. Default: 10.
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
		self.ndof = kwargs.pop("ndof", 10)

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
		bins_meas = np.linspace(
			self.range_meas[0], self.range_meas[1], self.nbins_meas + 1)

		# Get bin indices for each single event
		bin_index_true = np.digitize(true, bins=bins_true)
		bin_index_meas = np.digitize(meas, bins=bins_meas)

		# Build the response matrix A:
		# The entries of the response matrix A are integers.
		# Element aij of the matrix A is the number of MC events from bin j
		# of the true distribution x, which are measured in bin i of the
		# measured distributioin y.
		# So A has nbins_meas rows (index i) and nbins_true cols (index j)
		A = np.zeros((self.nbins_meas, self.nbins_true))
		# Loop over bin indices and count events
		for j in np.arange(0, self.nbins_true):
			for i in np.arange(0, self.nbins_meas):
				# np.digitize starts with 1 as first bin, so correct for that
				# by raising the bin indices
				mask = np.logical_and(
					bin_index_true == j+1,
					bin_index_meas == i+1,
					)
				# Sum over measured event weights to take the acceptance
				# into account
				A[i][j] = np.sum(meas_weights[mask])

		return A


	def predict(self):
		"""
		Use trained model on prediction data.
		"""



		return








