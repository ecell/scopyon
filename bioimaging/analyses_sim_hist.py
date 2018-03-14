#!/usr/bin/env python

import sys
import math
import copy
import csv

import numpy
import scipy
from array import array

from ROOT import *
from ROOT import gROOT

def get_data(data_array, flux_mW) :

        # read txt file
        gROOT.Reset()
	canvas = TCanvas()

	####
	psfile = './hist_%dmW.pdf' % (flux_mW)
	canvas.Print(psfile+"[")

	ntuple = TNtuple("data", "data", "I_true:x_true:y_true:I_reco:x_reco:y_reco:B_avg:B_dev")

	 # set data point  
	for item in data_array :  
            #item = data.split(',')
            ntuple.Fill(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7])

	Nbins = 40

	#### 1D histogram of photons
	hist_dx = TH1F("hist_dx", "%d mW : x_true - x_reco" % (flux_mW), Nbins, -4, 4)
	hist_dx.GetXaxis().SetTitle("x_true - x_reco [pixel]")
	#hist_dx.GetYaxis().SetTitle("Number")
	hist_dx.SetLineColor(1)
	hist_dx.SetLineWidth(2)
	#hist_dx.SetStats(0)
	ntuple.Draw("x_true-x_reco >> hist_dx")
	canvas.Print(psfile)

	#### 1D histogram of photons
	hist_dy = TH1F("hist_dy", "%d mW : y_true - y_reco" % (flux_mW), Nbins, -4, 4)
	hist_dy.GetXaxis().SetTitle("y_true - y_reco [pixel]")
	#hist_dy.GetYaxis().SetTitle("Number")
	hist_dy.SetLineColor(1)
	hist_dy.SetLineWidth(2)
	#hist_dy.SetStats(0)
	ntuple.Draw("y_true-y_reco >> hist_dy")
	canvas.Print(psfile)

	#### 1D histogram of photons
	hist_dr = TH1F("hist_dr", "%d mW : r_true - r_reco" % (flux_mW), Nbins, 0, 4)
	hist_dr.GetXaxis().SetTitle("r_true - r_reco [pixel]")
	#hist_dr.GetYaxis().SetTitle("Number")
	hist_dr.SetLineColor(1)
	hist_dr.SetLineWidth(2)
	#hist_dr.SetStats(0)
	ntuple.Draw("sqrt((x_true-x_reco)**2 + (y_true-y_reco)**2) >> hist_dr")
	canvas.Print(psfile)

	#### 1D histogram of photons
	hist_dI = TH1F("hist_dI", "%d mW : I_true - I_reco [counts]" % (flux_mW), Nbins, -4000, 6000)
	hist_dI.GetXaxis().SetTitle("I_true - I_reco [counts]")
	#hist_dI.GetYaxis().SetTitle("Number")
	hist_dI.SetLineColor(1)
	hist_dI.SetLineWidth(2)
	#hist_dI.SetStats(0)
	ntuple.Draw("I_true-I_reco >> hist_dI")
	canvas.Print(psfile)

	##########
	hist_2d_px = TH2F("hist_2d_px", "%d mW : Local precision [pixel] vs Intensity [counts]" % (flux_mW), Nbins, 0, 8000, Nbins, 0, 4)
	hist_2d_px.GetYaxis().SetTitle("Local precision [pixel]")
	hist_2d_px.GetXaxis().SetTitle("True Intensity [counts]")
	#hist_2d_px.SetStats(0)
	ntuple.Draw("sqrt((x_true-x_reco)**2 + (y_true-y_reco)**2):I_true >> hist_2d_px", "", "zcol")
	canvas.Print(psfile)

	##########
	hist_2d_nm = TH2F("hist_2d_nm", "%d mW : Local precision [nm] vs Intensity [counts]" % (flux_mW), Nbins, 0, 8000, Nbins, 0, 400)
	hist_2d_nm.GetYaxis().SetTitle("Local precision [nm]")
	hist_2d_nm.GetXaxis().SetTitle("True Intensity [counts]")
	#hist_2d_nm.SetStats(0)
	ntuple.Draw("sqrt((x_true-x_reco)**2 + (y_true-y_reco)**2)*73:I_true >> hist_2d_nm", "", "zcol")
	canvas.Print(psfile)

	##########
	hist_2d = TH2F("hist_2d", "%d mW : Local precision [nm] vs Intensity precision [counts]" % (flux_mW), Nbins, -4000, 4000, Nbins, 0, 400)
	hist_2d.GetYaxis().SetTitle("Local precision [nm]")
	hist_2d.GetXaxis().SetTitle("Intensity precision [counts]")
	#hist_2d.SetStats(0)
	ntuple.Draw("sqrt((x_true-x_reco)**2 + (y_true-y_reco)**2)*73:(I_true-I_reco) >> hist_2d", "", "zcol")
	canvas.Print(psfile)

	canvas.Print(psfile+']')


if __name__=='__main__':

    flux_mW = float(sys.argv[1])

    data = numpy.genfromtxt('./text_%dmW.dat' % (flux_mW))

    get_data(data, flux_mW)



