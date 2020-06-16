import pywt
import tikzplotlib
import matplotlib.pyplot as plt
from jaxlets.conv_fwt import get_filter_arrays
from jaxlets.conv_fwt_2d import construct_2d_filt

if __name__ == '__main__':
    wavelet = pywt.Wavelet('db2')
    # plot haar filter
    dec_lo, dec_hi, rec_lo, rec_hi = get_filter_arrays(wavelet, flip=True)
    plt.plot(dec_lo[0, :], '-.', label='dec_lo')
    plt.plot(dec_hi[0, :], '-.', label='dec_hi')
    plt.plot(rec_lo[0, :], '-.', label='rec_lo')
    plt.plot(rec_hi[0, :], '-.', label='rec_hi')
    plt.title(wavelet.name)
    plt.legend()
    tikzplotlib.save('db2_1d_plot.tex', standalone=True)
    plt.show()

    # plot haar 2d filters
    decfilt = construct_2d_filt(dec_lo, dec_hi)
    print(decfilt)
    fig=plt.figure()
    plt.axis('off')
    plt.title('2d analysis filters')
    fig.add_subplot(2, 2, 1)
    plt.imshow(decfilt[0, 0, :, :], vmin=-.5, vmax=.5)
    fig.add_subplot(2, 2, 2)
    plt.imshow(decfilt[1, 0, :, :], vmin=-.5, vmax=.5)
    fig.add_subplot(2, 2, 3)
    plt.imshow(decfilt[2, 0, :, :], vmin=-.5, vmax=.5)
    fig.add_subplot(2, 2, 4)
    plt.imshow(decfilt[4, 0, :, :], vmin=-.5, vmax=.5)

    tikzplotlib.save('db2_2d_analysis.tex', standalone=True)
    plt.show()
    print(decfilt.shape)

    recfilt = construct_2d_filt(rec_lo, rec_hi)
    print(recfilt)
    fig=plt.figure()
    plt.title('2d synthesis filters')
    plt.axis('off')

    fig.add_subplot(2, 2, 1)
    plt.imshow(recfilt[0, 0, :, :], vmin=-.5, vmax=.5)
    fig.add_subplot(2, 2, 2)
    plt.imshow(recfilt[1, 0, :, :], vmin=-.5, vmax=.5)
    fig.add_subplot(2, 2, 3)
    plt.imshow(recfilt[2, 0, :, :], vmin=-.5, vmax=.5)
    fig.add_subplot(2, 2, 4)
    plt.imshow(recfilt[4, 0, :, :], vmin=-.5, vmax=.5)
    tikzplotlib.save('db2_2d_synthesis.tex', standalone=True)
    plt.show()