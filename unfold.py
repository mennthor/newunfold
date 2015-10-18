#-*-coding:utf8-*-

class newunfold():
	"""
	Unfolding class. Structure oriented on scikit learn.

	The object gets the settings, like degrees of freedom, which define
	the unfolding model.

	The method fit trains the model on training data, eg. MC data.

	The method predict applies the trained model to test or real data.
	"""
	def __init__(self, ndof):
		self.ndof = ndof

	def fit(self):
		"""
		Fit model to training data.
		"""

	def predict(self):
		"""
		Use trained model on prediction data.
		"""