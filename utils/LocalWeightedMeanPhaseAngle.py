try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# Otherwise use the normal scipy fftpack ones instead (~2-3x slower!)
except ImportError:
    import warnings
    warnings.warn("""Module 'pyfftw' (FFTW Python bindings) could not be imported. To install it, try
                  running 'pip install pyfftw' from the terminal. Falling back on the slower
                  'fftpack' module for 2D Fourier transforms.""")
    from scipy.fftpack import fft2, ifft2

import numpy as np
from scipy.fftpack import ifftshift

def filtergrid(rows, cols):
    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)
    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)
    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)
    return radius, u1, u2

def _lowpassfilter(size, cutoff, n):
    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size
    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))


def rayleighmode(data, nbins=50):
    n, edges = np.histogram(data, nbins)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2.


def perfft2(im, compute_P=True, compute_spatial=False):
    if im.dtype not in ['float32', 'float64']:
        im = np.float64(im)
    rows, cols = im.shape
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + im[:, 0] - im[:, -1]
    s[:, -1] = s[:, -1] - im[:, 0] + im[:, -1]
    x, y = (2 * np.pi * np.arange(0, v) / float(v) for v in (cols, rows))
    cx, cy = np.meshgrid(x, y)
    denom = (2. * (2. - np.cos(cx) - np.cos(cy)))
    denom[0, 0] = 1.     # avoid / 0
    S = fft2(s) / denom
    S[0, 0] = 0      # enforce zero mean
    if compute_P or compute_spatial:

        P = fft2(im) - S

        if compute_spatial:
            s = ifft2(S).real
            p = im - s

            return S, P, s, p
        else:
            return S, P
    else:
        return S

def LwPA(img, nscale=5, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
                  k=2., cutOff=0.5, g=10., noiseMethod=-1, deviationGain=1.5):
  """RETURNS local weighted mean phase angle"""
  if img.dtype not in ['float32', 'float64']:
    img = np.float64(img)
    imgdtype = 'float64'
  else:
    imgdtype = img.dtype

  if img.ndim == 3:
    img = img.mean(2)

  rows, cols = img.shape
  epsilon = 1E-4          # used to prevent /0.
  _, IM = perfft2(img)     # periodic Fourier transform of image
  zeromat = np.zeros((rows, cols), dtype=imgdtype)
  sumAn = zeromat.copy()
  sumf = zeromat.copy()
  sumh1 = zeromat.copy()
  sumh2 = zeromat.copy()
  radius, u1, u2 = filtergrid(rows, cols)
  radius[0, 0] = 1.
  H = (1j * u1 - u2) / radius
    # filter parameters radius .45, 'sharpness' 15
  lp = _lowpassfilter((rows, cols), .45, 15)
  logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

  for ss in range(nscale):
    wavelength = minWaveLength * mult ** ss
    fo = 1. / wavelength  # Centre frequency of filter
    logRadOverFo = (np.log(radius / fo))
    logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
    logGabor *= lp      # Apply the low-pass filter
    logGabor[0, 0] = 0.  # Undo the radius fudge

    IMF = IM * logGabor   # Frequency bandpassed image
    f = np.real(ifft2(IMF))  # Spatially bandpassed image

    # Bandpassed monogenic filtering, real part of h contains
    # convolution result with h1, imaginary part contains
    # convolution result with h2.
    h = ifft2(IMF * H)
    h1, h2 = np.real(h), np.imag(h)
    # Amplitude of this scale component
    An = np.sqrt(f * f + h1 * h1 + h2 * h2)

        # Sum of amplitudes across scales
    sumAn += An
    sumf += f
    sumh1 += h1
    sumh2 += h2

        # At the smallest scale estimate noise characteristics from the
        # distribution of the filter amplitude responses stored in sumAn. tau
        # is the Rayleigh parameter that is used to describe the distribution.
    if ss == 0:
      # Use median to estimate noise statistics
      if noiseMethod == -1:
        tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))

      # Use the mode to estimate noise statistics
      elif noiseMethod == -2:
        tau = rayleighmode(sumAn.flatten())

      maxAn = An
    else:
            # Record the maximum amplitude of components across scales to
            # determine the frequency spread weighting
      maxAn = np.maximum(maxAn, An)

    width = (sumAn / (maxAn + epsilon) - 1.) / (nscale - 1)
    # Calculate the sigmoidal weighting function for this orientation
    weight = 1. / (1. + np.exp(g * (cutOff - width)))

    # fixed noise threshold
    if noiseMethod >= 0:
      T = noiseMethod
    else:
      totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))
      EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
      EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)
      # Noise threshold, must be >= epsilon
      T = np.max((EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon))

    # Final computation of key quantities
  ori = np.arctan(-sumh2 / sumh1)

    # Wrap angles between -pi and pi and convert radians to degrees
  ori = np.fix((ori % np.pi) / np.pi * 180.)

    # Feature type (a phase angle between -pi/2 and pi/2)
  ft = np.arctan2(sumf, np.sqrt(sumh1 * sumh1 + sumh2 * sumh2))

    # Overall energy
  energy = np.sqrt(sumf * sumf + sumh1 * sumh1 + sumh2 * sumh2)

    # Compute phase congruency.
  phase_dev = np.maximum(
        1. - deviationGain * np.arccos(energy / (sumAn + epsilon)), 0)
  energy_thresh = np.maximum(energy - T, 0)

  M = weight * phase_dev * energy_thresh / (energy + epsilon)

  return M, ori, ft, T

# Source LxPA functions: https://bit.ly/3qIntYA