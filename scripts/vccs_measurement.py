# -*- coding: utf-8 -*-


def Mag_V_L(I, RL, Ro, w, C):
	return I * (1/(1/RL + 1/Ro)) / (1 + (w * (1/(1/RL + 1/Ro)) * C)**2)**.5


def Mag_I_L(I, RL, Ro, w, C):
	I_L = 1 / RL * I / (1 / Ro + 1 / RL + 1j * w * C)
	return abs(I_L)


def Z_o_from_VLa_VLb(RLa, RLb, VLa, VLb):
    Z_o = (RLa * RLb * (VLb - VLa)) / (VLa * RLb - VLb * RLa)
    return Z_o


def Z_o_from_ILa_ILb(RLa, RLb, ILa, ILb):
	if abs(ILa - ILb) < 1E-12:
		return float("nan")
	Z_o = (ILb * RLb - ILa * RLa) / (ILa - ILb)
	return Z_o
