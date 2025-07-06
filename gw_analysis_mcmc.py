"""
Gravitational Wave Analysis using MCMC Parameter Estimation

This notebook demonstrates the analysis of gravitational wave data using 
Markov Chain Monte Carlo (MCMC) methods to estimate system parameters 
such as total mass and distance.

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.constants import G
import statistics as stat

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)

class GravitationalWaveAnalysis:
    """
    A class for analyzing gravitational wave data using MCMC parameter estimation.
    """
    
    def __init__(self):
        self.data_grav = None
        self.observed_data = None
        self.reference_data = None
        self.noise_mean = None
        self.noise_std = None
        self.interp_fn = None
        
    def load_data(self, gw_events_file, observed_file, reference_file):
        """
        Load gravitational wave data files.
        
        Parameters:
        -----------
        gw_events_file : str
            Path to gravitational wave events CSV file
        observed_file : str
            Path to observed waveform CSV file
        reference_file : str
            Path to reference waveform CSV file
        """
        # Load gravitational wave events data
        self.data_grav = pd.read_csv(gw_events_file).dropna()
        
        # Load observed waveform data
        self.observed_data = pd.read_csv(observed_file)
        
        # Load reference waveform data
        self.reference_data = pd.read_csv(reference_file)
        
        print(f"Loaded {len(self.data_grav)} gravitational wave events")
        print(f"Loaded {len(self.observed_data)} observed data points")
        print(f"Loaded {len(self.reference_data)} reference data points")
    
    def calculate_individual_masses(self, total_mass, chirp_mass):
        """
        Calculate individual masses from total mass and chirp mass.
        
        Parameters:
        -----------
        total_mass : array-like
            Total mass of the system
        chirp_mass : array-like
            Chirp mass of the system
            
        Returns:
        --------
        tuple
            Individual masses (m1, m2)
        """
        discriminant = total_mass**2 - 4 * (chirp_mass**(5/3)) * (total_mass**(1/3))
        m1 = (total_mass + np.sqrt(discriminant)) / 2
        m2 = total_mass - m1
        return m1, m2
    
    def plot_mass_distance_relation(self):
        """
        Plot the relationship between total mass and distance.
        """
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.data_grav['DL'], self.data_grav['Mtot'], 
                    xerr=self.data_grav['DL_err'], yerr=self.data_grav['Mtot_err'], 
                    fmt='*', capsize=5, alpha=0.7)
        plt.xlabel('Distance (Mpc)')
        plt.ylabel('Total Mass (M☉)')
        plt.title('Total Mass vs Distance from Earth')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def process_observed_data(self, gps_time_offset=1205951542.153363):
        """
        Process observed waveform data and estimate noise characteristics.
        
        Parameters:
        -----------
        gps_time_offset : float
            GPS time offset to convert to relative time
        """
        # Convert GPS time to relative time
        time_obs = self.observed_data['time (s)'] - gps_time_offset
        strain = self.observed_data['strain']
        
        # Plot original strain data
        plt.figure(figsize=(12, 4))
        plt.plot(time_obs, strain, alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.title('Gravitational Wave Strain vs Time')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Extract noise region (after the signal)
        noise_indices = np.where(time_obs > 0)[0]
        noise_strain = strain[noise_indices]
        
        # Calculate noise statistics
        self.noise_mean = np.mean(np.abs(noise_strain))
        self.noise_std = np.std(np.abs(noise_strain))
        
        print(f"Noise characteristics:")
        print(f"Mean: {self.noise_mean:.2e}")
        print(f"Standard deviation: {self.noise_std:.2e}")
        
        return time_obs, strain
    
    def setup_waveform_model(self, ref_mass=40, ref_distance=1):
        """
        Set up the gravitational waveform model using reference data.
        
        Parameters:
        -----------
        ref_mass : float
            Reference mass in solar masses
        ref_distance : float
            Reference distance in Mpc
        """
        self.ref_mass = ref_mass
        self.ref_distance = ref_distance
        
        # Create interpolation function from reference data
        self.interp_fn = interp1d(
            self.reference_data['time (s)'], 
            self.reference_data['strain'], 
            bounds_error=False, 
            fill_value=0
        )
        
        print(f"Waveform model setup with reference mass: {ref_mass} M☉, distance: {ref_distance} Mpc")
    
    def generate_waveform(self, mass, distance, time_array):
        """
        Generate gravitational waveform for given mass and distance.
        
        Parameters:
        -----------
        mass : float
            Total mass in solar masses
        distance : float
            Distance in Mpc
        time_array : array-like
            Time array for waveform generation
            
        Returns:
        --------
        tuple
            Time array and strain array
        """
        # Time scaling based on mass
        time_scaled = (self.ref_mass / mass) * time_array
        
        # Strain scaling based on mass and distance
        strain_scaled = (mass / self.ref_mass) * (self.ref_distance / distance) * self.interp_fn(time_scaled)
        
        return time_array, strain_scaled
    
    def log_likelihood(self, mass, distance, time_obs, strain_obs):
        """
        Calculate log-likelihood for given parameters.
        
        Parameters:
        -----------
        mass : float
            Total mass in solar masses
        distance : float
            Distance in Mpc
        time_obs : array-like
            Observed time array
        strain_obs : array-like
            Observed strain array
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        # Generate model waveform
        _, strain_model = self.generate_waveform(mass, distance, time_obs)
        
        # Calculate log-likelihood
        residuals = strain_obs - strain_model
        log_likelihood = -0.5 * np.sum((residuals**2) / (self.noise_std**2))
        
        return log_likelihood
    
    def run_mcmc(self, time_obs, strain_obs, initial_mass=77, initial_distance=1580, 
                 n_steps=10000, step_size=0.1):
        """
        Run MCMC parameter estimation.
        
        Parameters:
        -----------
        time_obs : array-like
            Observed time array
        strain_obs : array-like
            Observed strain array
        initial_mass : float
            Initial guess for mass
        initial_distance : float
            Initial guess for distance
        n_steps : int
            Number of MCMC steps
        step_size : float
            Step size for proposals
            
        Returns:
        --------
        tuple
            Mass and distance chains
        """
        # Initialize chains
        mass_chain = np.zeros(n_steps + 1)
        distance_chain = np.zeros(n_steps + 1)
        
        # Set initial values
        mass_chain[0] = initial_mass
        distance_chain[0] = initial_distance
        
        # MCMC loop
        for i in range(n_steps):
            # Current likelihood
            current_ll = self.log_likelihood(mass_chain[i], distance_chain[i], 
                                           time_obs, strain_obs)
            
            # Propose new parameters
            mass_proposed = mass_chain[i] + np.random.normal(0, step_size)
            distance_proposed = distance_chain[i] + np.random.normal(0, step_size)
            
            # Ensure positive values
            if mass_proposed <= 0 or distance_proposed <= 0:
                mass_chain[i + 1] = mass_chain[i]
                distance_chain[i + 1] = distance_chain[i]
                continue
            
            # Calculate proposed likelihood
            proposed_ll = self.log_likelihood(mass_proposed, distance_proposed, 
                                            time_obs, strain_obs)
            
            # Accept/reject step
            if proposed_ll > current_ll:
                # Accept
                mass_chain[i + 1] = mass_proposed
                distance_chain[i + 1] = distance_proposed
            else:
                # Metropolis criterion
                acceptance_prob = np.exp(proposed_ll - current_ll)
                if np.random.uniform() < acceptance_prob:
                    mass_chain[i + 1] = mass_proposed
                    distance_chain[i + 1] = distance_proposed
                else:
                    mass_chain[i + 1] = mass_chain[i]
                    distance_chain[i + 1] = distance_chain[i]
        
        return mass_chain, distance_chain
    
    def analyze_mcmc_results(self, mass_chain, distance_chain, burn_in=1000):
        """
        Analyze MCMC results and calculate statistics.
        
        Parameters:
        -----------
        mass_chain : array-like
            MCMC chain for mass
        distance_chain : array-like
            MCMC chain for distance
        burn_in : int
            Number of burn-in samples to discard
            
        Returns:
        --------
        dict
            Dictionary containing statistics
        """
        # Remove burn-in samples
        mass_samples = mass_chain[burn_in:]
        distance_samples = distance_chain[burn_in:]
        
        # Calculate statistics
        results = {
            'mass_mean': np.mean(mass_samples),
            'mass_std': np.std(mass_samples),
            'mass_median': np.median(mass_samples),
            'distance_mean': np.mean(distance_samples),
            'distance_std': np.std(distance_samples),
            'distance_median': np.median(distance_samples)
        }
        
        print("MCMC Results:")
        print(f"Mass: {results['mass_mean']:.2f} ± {results['mass_std']:.2f} M☉")
        print(f"Distance: {results['distance_mean']:.1f} ± {results['distance_std']:.1f} Mpc")
        
        return results
    
    def plot_mcmc_chains(self, mass_chain, distance_chain, burn_in=1000):
        """
        Plot MCMC chains for convergence analysis.
        
        Parameters:
        -----------
        mass_chain : array-like
            MCMC chain for mass
        distance_chain : array-like
            MCMC chain for distance
        burn_in : int
            Number of burn-in samples to highlight
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mass chain
        axes[0, 0].plot(mass_chain, alpha=0.7)
        axes[0, 0].axvline(burn_in, color='r', linestyle='--', alpha=0.7, label='Burn-in')
        axes[0, 0].set_xlabel('MCMC Step')
        axes[0, 0].set_ylabel('Mass (M☉)')
        axes[0, 0].set_title('Mass Chain')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Distance chain
        axes[0, 1].plot(distance_chain, alpha=0.7)
        axes[0, 1].axvline(burn_in, color='r', linestyle='--', alpha=0.7, label='Burn-in')
        axes[0, 1].set_xlabel('MCMC Step')
        axes[0, 1].set_ylabel('Distance (Mpc)')
        axes[0, 1].set_title('Distance Chain')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Mass histogram
        axes[1, 0].hist(mass_chain[burn_in:], bins=50, alpha=0.7, density=True)
        axes[1, 0].set_xlabel('Mass (M☉)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Mass Posterior Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distance histogram
        axes[1, 1].hist(distance_chain[burn_in:], bins=50, alpha=0.7, density=True)
        axes[1, 1].set_xlabel('Distance (Mpc)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distance Posterior Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_final_fit(self, time_obs, strain_obs, mass_best, distance_best):
        """
        Plot the final waveform fit.
        
        Parameters:
        -----------
        time_obs : array-like
            Observed time array
        strain_obs : array-like
            Observed strain array
        mass_best : float
            Best-fit mass
        distance_best : float
            Best-fit distance
        """
        # Generate best-fit waveform
        _, strain_fit = self.generate_waveform(mass_best, distance_best, time_obs)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_obs, strain_obs, 'b-', alpha=0.7, label='Observed Data')
        plt.plot(time_obs, strain_fit, 'r-', alpha=0.8, label='MCMC Best Fit')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.title(f'Gravitational Wave Fit (M = {mass_best:.1f} M☉, D = {distance_best:.0f} Mpc)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def calculate_physical_parameters(self, mass_solar, mass_ratio=1.0):
        """
        Calculate physical parameters from the estimated mass.
        
        Parameters:
        -----------
        mass_solar : float
            Total mass in solar masses
        mass_ratio : float
            Mass ratio (m2/m1)
            
        Returns:
        --------
        dict
            Dictionary containing physical parameters
        """
        # Convert to kg
        M_sun = 1.989e30  # kg
        total_mass_kg = mass_solar * M_sun
        
        # Calculate individual masses
        m1_kg = total_mass_kg / (1 + mass_ratio)
        m2_kg = mass_ratio * m1_kg
        
        # Calculate chirp mass
        chirp_mass_kg = ((m1_kg * m2_kg)**(3/5)) / ((m1_kg + m2_kg)**(1/5))
        chirp_mass_solar = chirp_mass_kg / M_sun
        
        results = {
            'total_mass_kg': total_mass_kg,
            'total_mass_solar': mass_solar,
            'mass1_kg': m1_kg,
            'mass2_kg': m2_kg,
            'chirp_mass_kg': chirp_mass_kg,
            'chirp_mass_solar': chirp_mass_solar
        }
        
        print("Physical Parameters:")
        print(f"Total Mass: {mass_solar:.2f} M☉")
        print(f"Individual Masses: {m1_kg/M_sun:.2f} M☉, {m2_kg/M_sun:.2f} M☉")
        print(f"Chirp Mass: {chirp_mass_solar:.2f} M☉")
        
        return results


def main():
    """
    Main analysis pipeline.
    """
    # Initialize analysis class
    gw_analysis = GravitationalWaveAnalysis()
    
    # Load data (replace with actual file paths)
    print("Loading data...")
    # gw_analysis.load_data('gravitationalwaveevents.csv', 
    #                      'Observedwaveform.csv', 
    #                      'reference_Mtot40Msun_Dist1Mpc.csv')
    
    # Process observed data
    print("Processing observed data...")
    # time_obs, strain_obs = gw_analysis.process_observed_data()
    
    # Setup waveform model
    print("Setting up waveform model...")
    # gw_analysis.setup_waveform_model()
    
    # Run MCMC
    print("Running MCMC analysis...")
    # mass_chain, distance_chain = gw_analysis.run_mcmc(time_obs, strain_obs)
    
    # Analyze results
    print("Analyzing results...")
    # results = gw_analysis.analyze_mcmc_results(mass_chain, distance_chain)
    
    # Plot results
    print("Plotting results...")
    # gw_analysis.plot_mcmc_chains(mass_chain, distance_chain)
    # gw_analysis.plot_final_fit(time_obs, strain_obs, 
    #                           results['mass_mean'], results['distance_mean'])
    
    # Calculate physical parameters
    # gw_analysis.calculate_physical_parameters(results['mass_mean'])
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
