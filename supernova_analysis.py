"""
Supernova Spectral Analysis - Sample Code from Thesis
=====================================================

This is a reconstructed sample of spectral analysis code from my thesis work.
Analyzes different types of supernovae: SLSN, SNIcn, and SNIbn.

Note: Original data files and complete codebase no longer accessible post-graduation.
This represents the core analysis methodology used in the research.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import pandas as pd
from astropy.io import ascii
from astropy import units as u
from astropy.modeling import models
from specutils.fitting import fit_generic_continuum, fit_lines
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.analysis import equivalent_width
from scipy.optimize import curve_fit

# =============================================================================
# SECTION 1: BASIC SPECTRAL PLOTTING AND COMPARISON
# =============================================================================

def plot_slsn_comparison():
    """
    Super-Luminous Supernova (SLSN) Analysis - PTF10nmn
    Multi-epoch spectral comparison with error bars
    """
    # Load spectral data for different epochs
    data0 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\PTF10nmn\PTF10nmn_2010-07-07.dat")
    data1 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\PTF10nmn\PTF10nmn_2011-03-04.dat")
    data2 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\PTF10nmn\PTF10nmn_2011-07-03.dat")
    data3 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\PTF10nmn\PTF10nmn_2012-02-20.dat")
    
    # Extract wavelength and flux data
    wavelength = data0['col1']
    flux = data0['col2']
    flux_err = data0['col3']
    
    wavelength1 = data1['col1']
    flux1 = data1['col2']
    flux_err1 = data1['col3']
    
    # Plot logarithmic flux comparison
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, np.log(flux), label='2010-07-07')
    plt.errorbar(wavelength1, np.log(flux1), label='2011-03-04')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('log(Flux)')
    plt.legend()
    plt.title('PTF10nmn - SLSN Temporal Evolution')
    plt.show()
    
    # Plot linear flux in specific wavelength range
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, flux, flux_err, label='2010-07-07')
    plt.errorbar(wavelength1, flux1, flux_err1, label='2011-03-04')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.xlim(5500, 6500)
    plt.legend()
    plt.title('PTF10nmn - Detailed View (5500-6500 Å)')
    plt.show()


def plot_snicn_comparison():
    """
    Type Ic-n Supernova Analysis - SN2019hgp
    Narrow emission line supernova spectral analysis
    """
    # Load spectral data
    cndata0 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\SN2019hgp\2019hgp_2019-06-08.dat")
    cndata1 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\SN2019hgp\2019hgp_2019-06-10.dat")
    
    wavelength = cndata0['col1']
    flux = cndata0['col2']
    wavelength1 = cndata1['col1']
    flux1 = cndata1['col2']
    
    # Logarithmic flux comparison
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, np.log(flux), label='2019-06-08')
    plt.errorbar(wavelength1, np.log(flux1), label='2019-06-10')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('log(Flux)')
    plt.legend()
    plt.title('SN2019hgp - Type Ic-n Evolution')
    plt.show()
    
    # Linear flux in blue region
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, flux, label='2019-06-08')
    plt.errorbar(wavelength1, flux1, label='2019-06-10')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.xlim(3000, 5000)
    plt.legend()
    plt.title('SN2019hgp - Blue Spectral Region')
    plt.show()


def plot_snibn_comparison():
    """
    Type Ibn Supernova Analysis - SN2011hw
    Narrow helium line supernova with strong CSM interaction
    """
    # Load spectral data
    bndata0 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\SN2011hw\SN_2011hw_2011-12-02.dat")
    bndata1 = ascii.read(r"C:\Users\SwiftX\Desktop\Summer Project\Data\SN2011hw\SN_2011hw_2011-12-26.dat")
    
    wavelength = bndata0['col1']
    flux = bndata0['col2']
    flux_err = bndata0['col3']
    
    wavelength1 = bndata1['col1']
    flux1 = bndata1['col2']
    flux_err1 = bndata1['col3']
    
    # Logarithmic flux without error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, np.log(flux))
    plt.errorbar(wavelength1, np.log(flux1))
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('log(Flux)')
    plt.title('SN2011hw - Type Ibn Logarithmic Flux')
    plt.show()
    
    # Linear flux with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, flux, flux_err, label='2011-12-02')
    plt.errorbar(wavelength1, flux1, flux_err1, label='2011-12-26')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.xlim(3000, 5000)
    plt.legend()
    plt.title('SN2011hw - Type Ibn Linear Flux')
    plt.show()


# =============================================================================
# SECTION 2: DETAILED ABSORPTION LINE ANALYSIS
# =============================================================================

def gaussian(x, a, mu, sigma):
    """
    Gaussian function for line profile fitting
    """
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


def analyze_calcium_lines():
    """
    Calcium H&K Line Analysis for SN2011hw
    Velocity profile analysis of Ca II H&K absorption lines
    """
    # Load high-resolution spectrum
    spectrum_file = r'C:\Users\SwiftX\Desktop\Summer Project\Data\SN2011hw\wop\SN_2011hw_2011-12-02.dat'
    
    try:
        with open(spectrum_file, 'r') as f:
            rows = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {spectrum_file}")
        return
    
    # Supernova redshift
    redshift = 0.123
    
    # Initialize arrays for Ca H and Ca K analysis
    wave1, flux1, vel_caH, vel_caK = [], [], [], []
    
    # Parse spectrum and extract Ca line region (3890-4000 Å rest frame)
    for row in rows:
        try:
            parts = row.split()
            rest_wavelength = float(parts[0]) / (1 + redshift)
            
            if 3890 < rest_wavelength < 4000:
                wave1.append(rest_wavelength)
                flux1.append(float(parts[1]))
                
                # Calculate velocities relative to Ca H (3934 Å) and Ca K (3969 Å)
                vel_caH.append((rest_wavelength - 3934) / 3934 * 300000)
                vel_caK.append((rest_wavelength - 3969) / 3969 * 300000)
        
        except (ValueError, IndexError):
            continue
    
    if not wave1:
        print("No data found in Ca line region")
        return
    
    # Create subplot for Ca line analysis
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    
    try:
        ax = plt.subplot(gs[0])
        
        # Continuum fitting and normalization
        spectrum = Spectrum1D(flux=flux1*u.dimensionless_unscaled, spectral_axis=wave1*u.AA)
        continuum_fit = fit_generic_continuum(spectrum)
        continuum_fitted = continuum_fit(wave1*u.AA)
        normalized_flux = np.array(flux1) / continuum_fitted.value
        
        # Plot velocity profiles
        plt.plot(vel_caH, normalized_flux, color='b', label='Ca H (3934 Å)')
        plt.plot(vel_caK, normalized_flux, color='r', label='Ca K (3969 Å)')
        
        # Formatting
        plt.legend(prop={'size': 16}, ncol=1, loc=4)
        plt.xlim(-1000, 1000)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Normalized Flux')
        plt.title('Ca II H&K Velocity Profiles')
        plt.grid(True, alpha=0.3)
        plt.savefig('2011hw_ca_velocity_profile.png', dpi=300, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error in Ca line analysis: {e}")
    
    plt.show()


def analyze_sodium_lines():
    """
    Sodium D Line Analysis for SN2011hw
    Velocity profile analysis of Na I D1 and D2 absorption lines
    """
    spectrum_file = r'C:\Users\SwiftX\Desktop\Summer Project\Data\SN2011hw\wop\SN_2011hw_2011-12-26.dat'
    
    try:
        with open(spectrum_file, 'r') as f:
            rows = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {spectrum_file}")
        return
    
    redshift = 0.123
    wave2, flux2, vel_NaD_1, vel_NaD_2 = [], [], [], []
    
    # Parse spectrum and extract Na D line region (5840-5960 Å rest frame)
    for row in rows:
        try:
            parts = row.split()
            rest_wavelength = float(parts[0]) / (1 + redshift)
            
            if 5840 < rest_wavelength < 5960:
                wave2.append(rest_wavelength)
                flux2.append(float(parts[1]))
                
                # Calculate velocities relative to Na D1 (5890 Å) and Na D2 (5896 Å)
                vel_NaD_1.append((rest_wavelength - 5890) / 5890 * 300000)
                vel_NaD_2.append((rest_wavelength - 5896) / 5896 * 300000)
        
        except (ValueError, IndexError):
            continue
    
    if not wave2:
        print("No data found in Na D line region")
        return
    
    try:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
        ax = plt.subplot(gs[1])
        
        # Continuum fitting and normalization
        spectrum = Spectrum1D(flux=flux2*u.dimensionless_unscaled, spectral_axis=wave2*u.AA)
        continuum_fit = fit_generic_continuum(spectrum)
        continuum_fitted = continuum_fit(wave2*u.AA)
        normalized_flux = np.array(flux2) / continuum_fitted.value
        
        # Plot velocity profiles
        plt.plot(vel_NaD_1, normalized_flux, color='b', label='Na D1 (5890 Å)')
        plt.plot(vel_NaD_2, normalized_flux, color='r', label='Na D2 (5896 Å)')
        
        # Formatting
        plt.legend(prop={'size': 16}, ncol=1, loc=4)
        plt.xlim(-1000, 1000)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Normalized Flux')
        plt.title('Na I D1&D2 Velocity Profiles')
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Error in Na line analysis: {e}")
    
    plt.show()


# =============================================================================
# SECTION 3: PHYSICAL PARAMETER CALCULATIONS
# =============================================================================

def calculate_shell_properties():
    """
    Calculate circumstellar shell properties from absorption line analysis
    """
    # Load processed spectral data
    data0 = ascii.read(r'C:\Users\SwiftX\Desktop\Summer Project\Data\SN2011hw\wop\SN_2011hw_2011-12-02.dat')
    data1 = ascii.read(r'C:\Users\SwiftX\Desktop\Summer Project\Data\SN2011hw\wop\SN_2011hw_2011-12-26.dat')
    
    wavelength = data0['col1']
    flux = data0['col2']
    wavelength1 = data1['col1']
    flux1 = data1['col2']
    
    # Plot spectral evolution with line identifications
    plt.figure(figsize=(12, 8))
    plt.errorbar(wavelength, np.log(flux), label='2011-12-02')
    plt.errorbar(wavelength1, np.log(flux1), label='2011-12-26')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('log(Flux)')
    plt.xlim(4000, 5000)
    plt.legend()
    plt.title('SN2011hw - Spectral Evolution with Line Identification')
    plt.show()
    
    # Calculate observed wavelengths for velocity components
    redshift = 0.123
    velocity_offset = -500  # km/s
    
    # Ca H and Ca K observed wavelengths for blueshifted component
    w1_caH = (((velocity_offset / 300000) * 3934) + 3934) * (1 + redshift)
    w2_caK = (((velocity_offset / 300000) * 3969) + 3969) * (1 + redshift)
    
    print(f"Ca H observed wavelength: {w1_caH:.2f} Å")
    print(f"Ca K observed wavelength: {w2_caK:.2f} Å")
    
    # Plot with line markers
    plt.figure(figsize=(12, 8))
    plt.errorbar(wavelength, np.log(flux), label='2011-12-02')
    plt.errorbar(wavelength1, np.log(flux1), label='2011-12-26')
    plt.axvline(x=w1_caH, color='red', linestyle='--', label='Ca H (-500 km/s)')
    plt.axvline(x=w2_caK, color='blue', linestyle='--', label='Ca K (-500 km/s)')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('log(Flux)')
    plt.xlim(4000, 5000)
    plt.legend()
    plt.title('SN2011hw - Velocity Component Identification')
    plt.show()
    
    # Calculate shell velocities
    delta_wave = w2_caK - w1_caH
    v_shell_caH = (delta_wave / 3934) * 300000
    v_shell_caK = (delta_wave / 3969) * 300000
    
    print(f"Shell velocity (Ca H): {v_shell_caH:.1f} km/s")
    print(f"Shell velocity (Ca K): {v_shell_caK:.1f} km/s")
    
    return v_shell_caH, v_shell_caK


def estimate_csm_mass():
    """
    Estimate circumstellar material (CSM) mass from absorption line properties
    """
    # Calculate shell velocities
    v_shell_caH, v_shell_caK = calculate_shell_properties()
    
    # Physical parameters
    optical_depth_caH = 0.0383  # Ca H line optical depth
    optical_depth_caK = 0.104   # Ca K line optical depth
    nova_velocity = 5600        # km/s, typical nova expansion velocity
    
    # Mass estimation formula (empirical relation)
    # M = 2×10^-8 × (v_shell/75)^2 × (v_nova/1000)^-2 × (τ/10) [solar masses]
    
    mass_caH = 2e-8 * (v_shell_caH / 75)**2 * (nova_velocity / 1000)**-2 * (optical_depth_caH / 10)
    mass_caK = 2e-8 * (v_shell_caK / 75)**2 * (nova_velocity / 1000)**-2 * (optical_depth_caK / 10)
    
    print(f"CSM mass estimate (Ca H): {mass_caH/1e-7:.2f} × 10^-7 solar masses")
    print(f"CSM mass estimate (Ca K): {mass_caK/1e-7:.2f} × 10^-7 solar masses")
    
    return mass_caH, mass_caK


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Supernova Spectral Analysis - Thesis Sample Code")
    print("=" * 50)
    print("Note: This is a reconstructed sample from thesis work.")
    print("Original data files are no longer accessible.\n")
    
    # Uncomment individual sections to run analysis
    
    # Section 1: Basic spectral plotting
    # plot_slsn_comparison()
    # plot_snicn_comparison()
    # plot_snibn_comparison()
    
    # Section 2: Detailed line analysis
    # analyze_calcium_lines()
    # analyze_sodium_lines()
    
    # Section 3: Physical calculations
    # calculate_shell_properties()
    # estimate_csm_mass()
    
    print("Analysis complete. Uncomment specific functions to run individual analyses.")
