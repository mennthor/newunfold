#-*-coding:utf8-*-

"""
Create root files with the sample MC data to use in TRUEE for comparison.
"""

import numpy as np
import ROOT
import mc_data_gen


N = 100000
xl = 0
xh = 2

mcd = mc_data_gen.LorentzianUnfoldData(N=N, range=[xl, xh])

true, meas = mcd.get_mc_sample()

f = ROOT.TFile("mc_flat.root", "recreate")
tree = ROOT.TTree("tree", "tree")

vals = np.zeros((2, 1))

# True and measured variable
tree.Branch("x1", vals[0], "x1/D")
tree.Branch("y", vals[1], "y/D")

for i in range(N):
	vals[0] = true["data"][i]
	vals[1] = meas["data"][i]
	tree.Fill()

tree.Write()
f.Close()


