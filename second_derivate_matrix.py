import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt

def create_second_derivate_matrix(n=5):
	"""
	Create the regularization matrix C iteratively from the commom
	submatrix cj. The regularization function is the second numerical
	derivation of the binned function f, written in quadratic form:
		(f''(x))^2 = sum_j (f''(xj))^2 = a^T C a
	where a=(a0,...,) is the vector of bin entries at bin position x
	"""
	if n < 3:
		print("Number of dimension must be >= 3.")
		return None
	# Whole n x n matrix
	C = np.zeros((n, n))
	# Submatrix for derivation around xj, 1<=j<=n-2 (last array index is n-1)
	cj = [[ 1 , -2 ,  1],
	      [-2 ,  4 , -2],
	      [ 1 , -2 ,  1]]
	# Put each submatrix in the big matrix and add overlapping entries
	for j in range(1, n-1):
		# Put in the submatrix at each sliced out 3x3 part around the
		# element diagonal at j
		C[j-1:j+2, j-1:j+2] += cj

	return C


if __name__ == '__main__':
	# print("n = 3 :\n{}\n".format(create_second_derivate_matrix(3)))
	# print("n = 4 :\n{}\n".format(create_second_derivate_matrix(4)))
	# print("n = 5 :\n{}\n".format(create_second_derivate_matrix(5)))
	# print("n = 6 :\n{}\n".format(create_second_derivate_matrix(6)))

	# Compare to continous function
	xl = 0
	xh = 1
	x = np.linspace(xl, xh, 100)
	f = lambda x: x**3
	f2_squ = lambda x: 6 * x
	f = lambda x: x**2
	f2_squ = lambda x: np.ones_like(x) * 2**2

	nbins = 5
	bins = np.linspace(xl, xh, nbins + 1)
	xj = 0.5 * (bins[:-1] + bins[1:])

	# Bin continous function by using the average function value between bins
	hist = np.zeros_like(xj)
	for j in range(len(xj)):
		hist[j] = scint.quad(f, bins[j], bins[j+1])[0] / (bins[j+1] - bins[j])

	# Plot continous and discrete function
	plt.plot(x, f(x))
	plt.errorbar(xj, hist, xerr=(bins[1]-bins[0])/2., fmt="|")
	# plt.show()

	# Compare second derivate square function (f''(x))^2
	C = create_second_derivate_matrix(n=nbins)
	print(C)
	print(hist)
	print(np.dot(hist, C))
	print(np.dot(C, hist))


	sec_der_squ_dis = np.dot(hist, np.dot(C, hist))
	sec_der_squ_con = scint.quad(f2_squ, xl, xh)[0]
	print("discrete (f'')^2    :  {}".format(sec_der_squ_dis))
	print("continously (f'')^2 :  {}".format(sec_der_squ_con))








