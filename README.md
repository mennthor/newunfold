# newunfold

Python unfolding implementation based on:
- Barlow, Beesten: Fitting using finite Monte Carlo samples (1993)
- Blobel: An unfolding method for high energy physics experiments (2002)


## Testdata

Data for building the response matrix and testing the unfolding is generated by the `lorentz` class. The generated data is the same as used by Blobel in his examples.

The pdf of the true distribution is the sum of three lorenz function normed correctly to unity:

	$$
	f_\text{true} = \sum_{i=1}^{3} b_k \frac{g_k^2}{(x-x_k^2) + g_k^2}
	$$