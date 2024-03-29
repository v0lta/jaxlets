# -*- coding: utf-8 -*-

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import jax.numpy as np
import pywt

from jaxlets.conv_fwt import wavedec, waverec
from jaxlets.lorenz import generate_lorenz


def test_haar_fwt_ifwt_16():
    # ---- Test harr wavelet analysis and synthesis on 16 sample signal. -----
    wavelet = pywt.Wavelet('haar')
    data = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.])
    data = np.expand_dims(np.expand_dims(data, 0), 0)
    coeffs = pywt.wavedec(data, wavelet, level=2)
    coeffs2 = wavedec(data, wavelet, level=2)
    cat_coeffs = np.concatenate(coeffs, -1)
    cat_coeffs2 = np.concatenate(coeffs2, -1)
    err = np.mean(np.abs(cat_coeffs - cat_coeffs2))
    assert err < 1e-4
    rest_data = waverec(coeffs2, wavelet)
    err = np.mean(np.abs(rest_data - data))
    assert err < 1e-4


def fwt_ifwt_lorenz(wavelet, mode='reflect'):
    # ---- Test wavelet analysis and synthesis on lorenz signal. -----
    lorenz = np.transpose(np.expand_dims(generate_lorenz(tmax=1.27)[:, 0], -1), [1, 0])
    data = np.expand_dims(lorenz, 0)
    coeff = wavedec(data, wavelet, mode=mode)
    pywt_coeff = pywt.wavedec(lorenz, wavelet, mode=mode)
    cat_coeff = np.concatenate(coeff, axis=-1)
    pywt_cat_coeff = np.concatenate(pywt_coeff, axis=-1)
    err = np.mean(np.abs(cat_coeff - pywt_cat_coeff))
    print("wavelet: {}, mode: {},    coefficient-error: {:2.2e}".format(
        wavelet.name, mode, err))
    # assert np.allclose(cat_coeff, pywt_cat_coeff, atol=1e-5, rtol=1e-4)
    rec_data = waverec(coeff, wavelet)
    err = np.mean(np.abs(rec_data - data))
    print("wavelet: {}, mode: {}, reconstruction-error: {:2.2e}".format(
        wavelet.name, mode, err))
    assert np.allclose(rec_data, data, atol=1e-5)


def test():
    for wavelet_str in ('haar', 'db2'):
        for boundary in ['constant', 'symmetric']:
            wavelet = pywt.Wavelet(wavelet_str)
            fwt_ifwt_lorenz(wavelet, mode=boundary)


if __name__ == '__main__':
    test()