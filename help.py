import numpy as np
import pylab as plt
import scipy.integrate as itg
import scipy.stats as st
import numpy.fft as ft
import numpy.random as rnd
import scipy.optimize as op
import scipy.special as sp
from astropy.io import fits

class Lightcurve(object):
    def __init__(self, time, flux, tbin, errors=None):
        self.time = time
        self.flux = flux
        self.errors = errors
        self.length = len(time)
        self.freq = np.arange(1, self.length / 2.0 + 1) / (self.length * tbin)
        self.psd = None
        self.mean = np.mean(flux)
        self.std_est = None
        self.std = np.std(flux)
        self.tbin = tbin
        self.fft = None
        self.periodogram = None
        self.pdfFit = None
        self.psdFit = None
        self.psdModel = None
        self.pdfModel = None
        self.bins = None

    def Simulate_DE_Lightcurve(self, PSDmodel=None, PSDinitialParams=None,
                               PSD_fit_method='Nelder-Mead', n_iter=1000,
                               PDFmodel=None, PDFinitialParams=None,
                               PDF_fit_method='BFGS', nbins=None,
                               RedNoiseL=100, aliasTbin=1, randomSeed=None,
                               maxIterations=1000, verbose=False, size=1,
                               LClength=None, histSample=False):
        if self.psdFit is None:
            if PSDmodel:
                self.Fit_PSD(initial_params=PSDinitialParams,
                             model=PSDmodel, fit_method=PSD_fit_method,
                             n_iter=n_iter, verbose=False)
            else:
                print("PSD not fitted, fitting using defaults (bending power law)...")
                self.Fit_PSD(verbose=False)

        if self.pdfFit is None and not histSample:
            if PDFmodel:
                self.Fit_PDF(initial_params=PDFinitialParams,
                             model=PDFmodel, fit_method=PDF_fit_method,
                             verbose=False, nbins=nbins)
            else:
                print("PDF not fitted, fitting using defaults (gamma + lognorm)")
                self.Fit_PDF(verbose=False)

        if self.psdFit is None or (self.pdfFit is None and not histSample):
            print("Simulation terminated due to failed fit(s)")
            return

        self.STD_Estimate(self.psdModel, self.psdFit['x'])

        if histSample:
            return Simulate_DE_Lightcurve(self.psdModel, self.psdFit['x'],
                                          size=size, LClength=LClength,
                                          lightcurve=self, histSample=True)
        else:
            return Simulate_DE_Lightcurve(self.psdModel, self.psdFit['x'],
                                          self.pdfModel, self.pdfFit['x'],
                                          size=size, LClength=LClength,
                                          lightcurve=self, histSample=False)

# ⬇️ This function is now properly outside the class
def Simulate_DE_Lightcurve(psd_model, psd_params,
                           pdf_model=None, pdf_params=None,
                           size=1, LClength=None, lightcurve=None, histSample=False):
    """
    Generate a synthetic light curve using a PSD model and PDF model.
    """
    if lightcurve is None:
        raise ValueError("A reference lightcurve must be provided.")

    if LClength is None:
        LClength = lightcurve.length

    dt = lightcurve.tbin
    N = LClength
    freqs = np.fft.rfftfreq(N, d=dt)

    psd_values = psd_model(freqs, *psd_params)
    amplitudes = np.sqrt(psd_values / 2.0)

    random_phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    complex_spectrum = amplitudes * (np.cos(random_phases) + 1j * np.sin(random_phases))

    time_series = np.fft.irfft(complex_spectrum, n=N)
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)

    if histSample and pdf_model is None:
        sorted_ts = np.sort(time_series)
        sorted_flux = np.sort(lightcurve.flux)
        ranks = np.argsort(np.argsort(time_series))
        time_series = sorted_flux[ranks]
    elif pdf_model is not None and pdf_params is not None:
        cdf_vals = st.norm.cdf(time_series)
        transformed = pdf_model.ppf(cdf_vals, *pdf_params)
        time_series = transformed

    time_array = np.arange(N) * dt
    simulated_lc = Lightcurve(time_array, time_series, dt)

    return simulated_lc

def Load_Lightcurve(fileroute, tbin,
                    time_col=None, flux_col=None, error_col=None,
                    extension=1, element=None):
    two = False
    print(f"\nLoading lightcurve from {fileroute}...")

    if time_col is not None and flux_col is not None:
        print("Using FITS loader (with columns)")
        ...
    elif time_col or flux_col:
        print("*** LOAD FAILED ***")
        print("Both time_col and flux_col must be defined for FITS input.")
        return
    else:
        print("Using plain text loader...")
        try:
            time, flux, error = [], [], []
            success = False
            read = 0

            with open(fileroute, 'r') as file:
                for i, line in enumerate(file):
                    line = line.strip()
                    columns = line.split(',') if ',' in line else line.split()

                    if i == 0 and not columns[0].replace('.', '', 1).isdigit():
                        print("Skipping header:", columns)
                        continue

                    print(f"Line {i}: {columns}")

                    if len(columns) == 3:
                        t_val, f_val, e_val = columns
                        try:
                            time.append(float(t_val))
                            flux.append(float(f_val))
                            error.append(float(e_val))
                            read += 1
                            success = True
                        except ValueError:
                            print(f"ValueError on line {i} - skipping")
                    elif len(columns) == 2:
                        t_val, f_val = columns
                        try:
                            time.append(float(t_val))
                            flux.append(float(f_val))
                            read += 1
                            success = True
                            two = True
                        except ValueError:
                            print(f"ValueError on line {i} - skipping")

            if success:
                print(f"✅ Successfully read {read} lines of data")
            else:
                print("*** LOAD FAILED ***\nText file must have 2 or 3 numeric columns")
                return

        except IOError:
            print("*** LOAD FAILED ***\nInput file not found")
            return

    print("Creating Lightcurve object...")
    if two:
        return Lightcurve(np.array(time), np.array(flux), tbin)
    else:
        return Lightcurve(np.array(time), np.array(flux), tbin, np.array(error))