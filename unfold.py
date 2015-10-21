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
		Define model parameters

		Parameters
		----------
		ndof : float
			Degress of freedom for the regularization
		nbins_true : int
			Number of bins for the true MC trainig data
		nbins_meas : int
			Number of bins for the measured MC training data
		nbins_meas : int
			Number of bins for the predicted unfolded data
		range_true : ndarray
			Range for the binning of the true MC training data
		range_meas : ndarray
			Range for the binning of the measured MC training data
		"""
		self.ndof = kwargs.pop("ndof", 10)
		self.nbins_true = kwagrs.pop("nbins_true", 20)
		self.nbins_meas = kwagrs.pop("nbins_meas", 20)
		self.nbins_pred = kwagrs.pop("nbins_pred", 20)


	def fit(self, true, meas):
		"""
		Fit model to training data.

		Parameters
		----------
		true : ndarray
			One dimensional true MC data used to build the response matrix.
		measured :
			One dimensional measured MC data used to build the response matrix.
		"""
		# Track bin content on event base to build the response matrix






	def predict(self):
		"""
		Use trained model on prediction data.
		"""