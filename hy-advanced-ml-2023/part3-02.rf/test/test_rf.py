#!/usr/bin/env python3

import sys
import unittest
from tmc import points
from tmc.utils import load, get_out

import numpy as np
from numpy.random import choice as orig_choice # need an own import, otherwise it will get patched

from unittest.mock import patch

from .utils import errormsg, ApproxTest



def split_count(T):
	if type(T) is tuple:
		return 1 + split_count(T[1]) + split_count(T[2])
	return 0

def get_par(call, pos, name, default):
	args, kwargs = call
	if len(args) > pos:
		return args[pos]
	else:
		return kwargs.get(name, default)
	

@points('3.2.1', '3.2.2', '3.2.3', '3.2.4', '3.2.5')
class RfTester(ApproxTest):

	def analyze_mock(self, choice_mock, rsize, nrsize):
		rcount = 0
		nrcount = 0
		rvalid = 0
		nrvalid = 0

		for call in choice_mock.call_args_list:
			replace = get_par(call, 2, 'replace', True)
			size = get_par(call, 1, 'size', None)
			array = get_par(call, 0, 'a', 0)

			if size:
				size = np.prod(size)
			else:
				size = 1

			if type(array) is int:
				a = array
			else:
				a = len(array)


			if replace:
				rcount += 1
				if size == rsize:
					rvalid += 1
			else:
				nrcount += 1
				if size == min(nrsize, a):
					nrvalid += 1

		return rcount, nrcount, rvalid, nrvalid


	@patch('numpy.random.choice', side_effect=orig_choice)
	def test_rf(self, choice_mock):
		rf = load('src.rf', 'rf')
		fit = load('src.rf', 'fit')

		D = np.loadtxt('test/train.txt')
		ltrain = D[:,0]
		Xtrain = D[:,1:]

		D = np.loadtxt('test/test.txt')
		ltest = D[:,0]
		Xtest = D[:,1:]
		cnt, k = Xtrain.shape

		np.random.seed(2022)
		p, misclass = rf(Xtrain, ltrain, 10, cnt, int(np.sqrt(k)), Xtest, ltest)

		rcount, nrcount, rvalid, nrvalid = self.analyze_mock(choice_mock, cnt, int(np.sqrt(k)))

		self.assertEqual(rcount, 10, "number of choice calls with replacement doesn't match treecnt")
		self.assertEqual(rvalid, rcount, "choice is called with incorrect size, you should sample samplecnt rows")

		misclass1 = np.array([0.192, 0.278, 0.132, 0.16,  0.138, 0.146, 0.122, 0.134, 0.12, 0.136])
		p1 = np.loadtxt('test/votes.txt')

		self.assertApprox(misclass, misclass1, errormsg("Incorrect misclassification rate (Xtrain = train.txt, Xtest = test.txt)", misclass1, misclass))
		self.assertApprox(p, p1, errormsg("Incorrect p (Xtrain = train.txt, Xtest = test.txt)", p1, p))




	@patch('numpy.random.choice', side_effect=orig_choice)
	def test_fit(self, choice_mock):
		rf = load('src.rf', 'rf')
		fit = load('src.rf', 'fit')

		D = np.loadtxt('test/toy.txt')
		l = D[:,0]
		X = D[:,1:]

		np.random.seed(2022)
		tree = fit(X, l, 2)

		rcount, nrcount, rvalid, nrvalid = self.analyze_mock(choice_mock, 0, 2)

		self.assertEqual(nrcount, split_count(tree), "number of choice calls without replacement doesn't match the number of splits")
		self.assertEqual(nrvalid, nrcount, "choice is called with incorrect size, you should sample at most featurecnt features")

		print(tree)
		t1 = (2, (3, (1, (0, -1, -1), (0, -1, -1)), (1, (0, -1, -1), (0, -1, -1))), (1, (0, (3, -1, -1), (3, -1, -1)), (3, (0, 1, 1), (0, 1, 1))))
		self.assertEqual(tree, t1, "Fit doesn't return correct tree (D = toy.txt, featurecnt=2, random state = 2022)")
