# Supernova Spectral Analysis - Thesis Sample Code

This repository contains a reconstructed sample of the spectral analysis code from my thesis work on supernova classification and circumstellar material interaction.

## Background

**Note:** This is a reconstructed sample from my graduate thesis. The original codebase and data files are no longer accessible post-graduation. This represents the core methodology and analysis approach used in the research.

## Research Focus

The code analyzes spectral data from three types of supernovae:
- **SLSN (Super-Luminous Supernovae)**: PTF10nmn
- **SNIcn (Type Ic-n Supernovae)**: SN2019hgp  
- **SNIbn (Type Ibn Supernovae)**: SN2011hw

## Code Structure

### Section 1: Basic Spectral Analysis
- Multi-epoch spectral comparisons
- Logarithmic and linear flux plotting
- Temporal evolution analysis

### Section 2: Absorption Line Analysis
- Calcium H&K line velocity profiles
- Sodium D line analysis
- Continuum fitting and normalization
- Velocity profile extraction

### Section 3: Physical Parameter Calculations
- Shell velocity measurements
- Circumstellar material mass estimates
- Line identification and marking

## Dependencies

```python
numpy
matplotlib
pandas
astropy
specutils
scipy
```

## Usage

```python
# Run individual analysis sections by uncommenting in main:

# Basic spectral plotting
plot_slsn_comparison()
plot_snicn_comparison() 
plot_snibn_comparison()

# Detailed line analysis
analyze_calcium_lines()
analyze_sodium_lines()

# Physical calculations
calculate_shell_properties()
estimate_csm_mass()
```

## Key Scientific Methods

1. **Spectral Data Processing**: Loading and handling multi-epoch astronomical spectra
2. **Continuum Normalization**: Using specutils for automated continuum fitting
3. **Velocity Profile Analysis**: Converting wavelength shifts to velocity space
4. **Physical Parameter Extraction**: Calculating shell velocities and mass estimates

## Sample Results

The code produces:
- Spectral evolution plots showing temporal changes
- Velocity profile plots for absorption lines
- Physical parameter estimates for circumstellar material

## Limitations

- Data paths are preserved from original thesis work but files are no longer available
- Some sections may require adjustment for different data formats
- This represents methodology rather than a complete working pipeline

## Research Context

This work was part of a thesis investigating the interaction between supernova ejecta and circumstellar material, particularly in Type Ibn supernovae which show strong narrow helium lines indicative of dense circumstellar shells.

---

*This code sample demonstrates the analytical methods used in graduate-level astronomical research on supernova spectroscopy and circumstellar material interaction.*