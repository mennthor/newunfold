#-*-coding:utf8-*-

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

		self.range_true = self.kwargs.pop("range_true", [0, 1])
		self.range_meas = self.kwargs.pop("range_meas", [0, 1])
		self.range_pred = self.kwargs.pop("range_pred", [0, 1])

		self.nbins_true = kwagrs.pop("nbins_true", 10)
		self.nbins_meas = kwagrs.pop("nbins_meas", 10)
		self.nbins_pred = kwagrs.pop("nbins_pred", 10)


	def fit(self, true, meas):
		"""
		Fit model to training data from MC.

		True and measured MC data are used in pairs (true[i], meas[i])
		to map the deetector influence.

		Parameters
		----------
		true : ndarray
			One dimensional unbinned true MC data used to build the
			response matrix.
		measured :
			One dimensional unbinned measured MC data used to build the
			response matrix.
		"""
		# Track bin content on event base to build the response matrix
		bins_true = np.linspace(
			self.range_true[0], self.range_true[1], nbins_true)
		bins_meas = np.linspace(
			self.range_meas[0], self.range_meas[1], nbins_meas)

		bin_index_by_event_true = np.digitze(true, bins=nbins_true)
		bin_index_by_event_meas = np.digitze(meas, bins=nbins_meas)

		return


	def predict(self):
		"""
		Use trained model on prediction data.
		"""

		return








