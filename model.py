"""
Cosmological Time-Scaling Model: Numerical Pipeline
==========================================================

This module implements the pipeline for testing the time-scaling model.

The pipeline includes:
1. Symbolic framework definition
2. Observational data fitting (Supernovae, GRB, Cosmic Chronometers, CMB)
3. Scalar field dynamics simulation
4. Statistical model comparison with MCMC support

Date: 2025 (Artur Chudzik)
Link: https://github.com/archudzik/timeScalingModel

"""

import os
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp, quad
import sympy as sp
import warnings
import emcee
from scipy.optimize import differential_evolution

random_number = 1234
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
np.random.seed(random_number)


class CosmologicalFramework:
    """
    Symbolic framework for cosmological equations and relationships.
    Defines the mathematical foundation of the time-scaling model.
    """

    def __init__(self):
        self.setup_symbols()
        self.derive_equations()

    def setup_symbols(self):
        """Initialize symbolic variables"""
        self.t, self.t0 = sp.symbols('t t0', positive=True)

        self.alpha = sp.Function('alpha')(self.t)
        self.alpha_dot = sp.diff(self.alpha, self.t)
        self.alpha_2dot = sp.diff(self.alpha, self.t, 2)

        self.G = sp.symbols('G', positive=True)
        self.c = sp.symbols('c', positive=True)

    def derive_equations(self):
        """Derive key cosmological relationships"""
        # Scale factor
        self.a = (self.t / self.t0)**self.alpha
        self.a_dot = sp.diff(self.a, self.t)
        self.a_2dot = sp.diff(self.a_dot, self.t)
        self.a_3dot = sp.diff(self.a_2dot, self.t)

        # Hubble parameter
        # For time-varying alpha: H = (α + α̇*ln(t/t₀))/t
        self.H = (self.alpha + self.alpha_dot *
                  sp.log(self.t / self.t0)) / self.t

        # Energy density from first Friedmann equation
        self.rho = sp.simplify((3 * self.H**2) / (8 * sp.pi * self.G))

        # Pressure from second Friedmann equation
        # ä/a = -4πG(ρ + 3p/c²)/3
        lhs = self.a_2dot / self.a
        self.p = sp.simplify((-(lhs * 3) / (4 * sp.pi * self.G)) - self.rho)

        # Equation of state parameter
        self.w = sp.simplify(self.p / (self.rho * self.c**2))

        # Deceleration parameter
        self.q = sp.simplify(-self.a * self.a_2dot / self.a_dot**2)

        # Jerk parameter
        self.jerk = sp.simplify(self.a**2 * self.a_3dot / self.a_dot**3)

    def print_equations(self):
        print("=== Cosmological Framework Equations ===")
        print(f"Scale factor a(t): {self.a}")
        print(f"Hubble parameter H(t): {self.H}")
        print(f"Energy density ρ(t): {self.rho}")
        print(f"Pressure p(t): {self.p}")
        print(f"Equation of state w(t): {self.w}")
        print(f"Deceleration parameter q(t): {self.q}")
        print(f"Jerk parameter j(t): {self.jerk}")


class ObservationalDataAnalyzer:
    """
    Handles fitting cosmological models to observational data
    """

    def __init__(self):
        self.c = 299792.458  # km/s
        self.r_s = 147.78    # Mpc (sound horizon)
        self.M_B = -19.3  # Absolute magnitude
        self.sn_M_B = 25 + self.M_B  # SN Absolute magnitude
        self.grb_M_B = 25  # GRB Absolute magnitude
        self.cmb_M_B = 5  # CMB Absolute magnitude

        self.bounds_lcdm = [(65, 75), (0, 0.5)]  # H0, Omega_m
        self.bounds_ts = [(70, 75), (1, 2)]  # H0, alpha

        self.bounds_lcdm_cmb = [(65, 75), (0, 20)]  # H0, mu0
        self.bounds_ts_cmb = [(70, 75), (1, 2), (0, 20)]  # H0, alpha, mu0

        self.h_planck_measured = 67.4

        # Model results storage
        self.results = {}
        self.cmb_data = {}

    def load_planck_tt_spectrum(self, file_path):
        """
        Load and properly process Planck TT power spectrum data
        """
        ell_vals = []
        cl_vals = []
        cl_err_vals = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        ell_vals.append(float(parts[0]))
                        cl_vals.append(float(parts[1]))
                        # If error column exists, use it; otherwise estimate
                        if len(parts) >= 3:
                            cl_err_vals.append(float(parts[2]))
                        else:
                            # Estimate error as ~5% of signal for high-ell, higher for low-ell
                            if float(parts[0]) < 30:
                                cl_err_vals.append(0.1 * float(parts[1]))
                            else:
                                cl_err_vals.append(0.05 * float(parts[1]))

        ell_vals = np.array(ell_vals)
        cl_vals = np.array(cl_vals)
        cl_err_vals = np.array(cl_err_vals)

        # Convert to D_ell = ell(ell+1)C_ell/(2π)
        Dl_vals = ell_vals * (ell_vals + 1) * cl_vals / (2 * np.pi)
        Dl_err_vals = ell_vals * (ell_vals + 1) * cl_err_vals / (2 * np.pi)
        return ell_vals, cl_vals, Dl_vals, Dl_err_vals

    def load_real_data(
        self,
        path_cc="./dataset/Chronometers.txt",
        path_supernova="./dataset/Pantheon+SH0ES.csv",
        path_grb="./dataset/GRB.txt",
        path_cmb="./dataset/Planck-TT.txt"
    ):
        """
        Load real observational data including enhanced CMB processing
        """

        cc_data = np.loadtxt(path_cc)
        z_cc = cc_data[:, 0]
        H_cc = cc_data[:, 1]
        H_err = cc_data[:, 2]
        self.cc_data = {'z': z_cc, 'H': H_cc, 'H_err': H_err}

        pantheon_data = pd.read_csv(path_supernova)
        pantheon_data = pantheon_data[[
            'zHD', 'm_b_corr', 'm_b_corr_err_DIAG']].dropna()
        pantheon_data.columns = ['z', 'mu', 'mu_err']
        z_sn = pantheon_data['z'].values
        mu_obs = pantheon_data['mu'].values
        mu_err = pantheon_data['mu_err'].values
        self.sn_data = {'z': z_sn, 'mu': mu_obs, 'mu_err': mu_err}

        grb_data_fixed = pd.read_csv(path_grb, sep=' ')
        grb_data_clean = grb_data_fixed[['Redshift', 'mu', 'sigma_mu']].copy()
        grb_data_clean.columns = ['z', 'mu', 'sigma_mu']
        self.grb_data = grb_data_clean

        # Enhanced CMB data loading
        ell_vals, cl_vals, Dl_vals, Dl_err_vals = self.load_planck_tt_spectrum(
            path_cmb)
        self.cmb_data = {
            'ell': ell_vals,
            'Dl': Dl_vals,
            'Dl_err': Dl_err_vals,
            'ell_full': ell_vals,
            'Dl_full': Dl_vals,
            'Dl_err_full': Dl_err_vals,
            'cl_vals': cl_vals
        }
        print(f"Loaded: {len(self.cc_data['z'])} CC, {len(self.sn_data['z'])} SN, "
              f"{len(self.grb_data)} GRB, {len(self.cmb_data['ell'])} CMB points")

    def _distance_modulus_LCDM(self, z, H0, Omega_m, M_B):
        """Calculate distance modulus for LCDM model"""
        def integrand(z_val):
            return self.c / (H0 * np.sqrt(Omega_m * (1 + z_val)**3 + (1 - Omega_m)))

        DL = []
        for zi in z:
            integral = quad(integrand, 0, zi)[0]
            DL.append((1 + zi) * integral)

        DL = np.array(DL)
        return 5 * np.log10(DL) + M_B

    def _distance_modulus_time_scaling(self, z, H0, alpha, M_B):
        """Calculate distance modulus for time-scaling model"""
        def integrand(z_val):
            return self.c / (H0 * (1 + z_val)**(1/alpha))

        DL = []
        for zi in z:
            integral = quad(integrand, 0, zi)[0]
            DL.append((1 + zi) * integral)

        DL = np.array(DL)
        return 5 * np.log10(DL) + M_B

    def _H_LCDM(self, z, H0, Omega_m):
        """Hubble parameter for LCDM"""
        return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

    def _H_time_scaling(self, z, H0, alpha):
        """Hubble parameter for time-scaling model"""
        return H0 * (1 + z)**(1.0 / alpha)

    def _DM_over_rs_LCDM(self, z, H0, Omega_m):
        """Angular diameter distance over sound horizon for LCDM"""
        def integrand(z_val):
            return self.c / (H0 * np.sqrt(Omega_m * (1 + z_val)**3 + (1 - Omega_m)))

        DM = []
        for zi in z:
            integral = quad(integrand, 0, zi)[0]
            DM.append(integral / (1 + zi))  # Angular diameter distance

        return np.array(DM) / self.r_s

    def _DM_over_rs_time_scaling(self, z, H0, alpha):
        """Angular diameter distance over sound horizon for time-scaling"""
        def integrand(z_val):
            return self.c / (H0 * (1 + z_val)**(1/alpha))
        DM = []
        for zi in z:
            integral = quad(integrand, 0, zi)[0]
            DM.append(integral / (1 + zi))  # Angular diameter distance
        return np.array(DM) / self.r_s

    def _fit_supernovae(self):
        """Fit models to supernova data (original implementation)"""
        z, mu_obs, mu_err = self.sn_data['z'], self.sn_data['mu'], self.sn_data['mu_err']

        def chi2_lcdm(params):
            H0, Om = params
            mu_model = self._distance_modulus_LCDM(z, H0, Om, self.sn_M_B)
            return np.sum(((mu_obs - mu_model) / mu_err)**2)

        def chi2_time_scaling(params):
            H0, alpha = params
            mu_model = self._distance_modulus_time_scaling(
                z, H0, alpha, self.sn_M_B)
            return np.sum(((mu_obs - mu_model) / mu_err)**2)

        res_lcdm = differential_evolution(
            chi2_lcdm, bounds=self.bounds_lcdm)
        res_ts = differential_evolution(
            chi2_time_scaling, bounds=self.bounds_ts)

        n_data = len(z)
        self.results['SN'] = {
            'LCDM': {
                'params': res_lcdm.x,
                'chi2': res_lcdm.fun,
                'AIC': res_lcdm.fun + 2*2,
                'BIC': res_lcdm.fun + 2*np.log(n_data),
            },
            'Time-Scaling': {
                'params': res_ts.x,
                'chi2': res_ts.fun,
                'AIC': res_ts.fun + 2*2,
                'BIC': res_ts.fun + 2*np.log(n_data),
            }
        }

    def _fit_grb(self):
        """Fit GRB-derived distance modulus data"""
        z = self.grb_data['z'].values
        mu_obs = self.grb_data['mu'].values
        mu_err = self.grb_data['sigma_mu'].values

        def chi2_lcdm(params):
            H0, Om = params
            mu_model = self._distance_modulus_LCDM(z, H0, Om, self.grb_M_B)
            return np.sum(((mu_obs - mu_model) / mu_err) ** 2)

        def chi2_ts(params):
            H0, alpha = params
            mu_model = self._distance_modulus_time_scaling(
                z, H0, alpha, self.grb_M_B)
            return np.sum(((mu_obs - mu_model) / mu_err) ** 2)

        res_lcdm = differential_evolution(chi2_lcdm, bounds=self.bounds_lcdm)
        res_ts = differential_evolution(chi2_ts, bounds=self.bounds_ts)

        n_data = len(z)
        self.results['GRB'] = {
            'LCDM': {
                'params': res_lcdm.x,
                'chi2': res_lcdm.fun,
                'AIC': res_lcdm.fun + 2 * 2,
                'BIC': res_lcdm.fun + 2 * np.log(n_data)
            },
            'Time-Scaling': {
                'params': res_ts.x,
                'chi2': res_ts.fun,
                'AIC': res_ts.fun + 2 * 2,
                'BIC': res_ts.fun + 2 * np.log(n_data)
            }
        }

    def _fit_cosmic_chronometers(self):
        """Fit models to cosmic chronometer data"""
        z_cc = self.cc_data['z']
        H_obs = self.cc_data['H']
        H_err = self.cc_data['H_err']

        def chi2_lcdm(params):
            H0, Om = params
            H_model = self._H_LCDM(z_cc, H0, Om)
            return np.sum(((H_obs - H_model) / H_err)**2)

        def chi2_time_scaling(params):
            H0, alpha = params
            H_model = self._H_time_scaling(z_cc, H0, alpha)
            return np.sum(((H_obs - H_model) / H_err)**2)

        res_lcdm = differential_evolution(chi2_lcdm, bounds=self.bounds_lcdm)
        res_ts = differential_evolution(
            chi2_time_scaling, bounds=self.bounds_ts)

        n_data = len(z_cc)
        self.results['CC'] = {
            'LCDM': {'params': res_lcdm.x, 'chi2': res_lcdm.fun,
                     'AIC': res_lcdm.fun + 2*2, 'BIC': res_lcdm.fun + 2*np.log(n_data)},
            'Time-Scaling': {'params': res_ts.x, 'chi2': res_ts.fun,
                             'AIC': res_ts.fun + 2*2, 'BIC': res_ts.fun + 2*np.log(n_data)}
        }

    def _fit_cmb(self):
        """Fit simplified models to transformed Planck D_ell spectrum using μ_CMB = log10(D_ell)"""

        # Approximate redshift mapping for multipoles
        ell = self.cmb_data['ell']
        Dl = self.cmb_data['Dl']
        Dl_err = self.cmb_data['Dl_err']
        valid = Dl > 0

        # Filter all arrays
        ell = ell[valid]
        Dl = Dl[valid]
        Dl_err = Dl_err[valid]

        # Brute-force optimization for completeness:
        def error_kappa(kappa):
            ell_peak = 220
            z_target = 1089
            z_eff = kappa * ell_peak
            return abs(z_eff - z_target)

        # Minimize absolute error between z_eff and z_target
        result_kappa = minimize_scalar(
            error_kappa, bounds=(1, 10), method='bounded')
        best_kappa = result_kappa.x
        print(f"best_kappa={best_kappa}")

        # Redshift mapping and transformed modulus
        z_vals = ell * best_kappa
        mu_cmb = np.log10(Dl)
        mu_err_prop = np.abs(0.434 * Dl_err / Dl)
        mu_err = np.minimum(mu_err_prop, 0.1 * mu_cmb)

        def lcdm_model(z, H0, mu0):
            return np.log10(1 + z) + np.log10(H0 / self.h_planck_measured) + mu0

        def timescaling_model(z, H0, alpha, mu0):
            return np.log10(1e3 / (1 + z) ** (1 - alpha) * (H0 / self.h_planck_measured)) + mu0

        def chi2_lcdm(p):
            H0, mu0 = p
            model = lcdm_model(z_vals, H0, mu0)
            return np.sum(((mu_cmb - model) / mu_err) ** 2)

        def chi2_ts(p):
            H0, alpha, mu0 = p
            model = timescaling_model(z_vals, H0, alpha, mu0)
            return np.sum(((mu_cmb - model) / mu_err) ** 2)

        res_lcdm = minimize(
            chi2_lcdm, x0=[self.h_planck_measured, 5.0], bounds=self.bounds_lcdm_cmb, method='L-BFGS-B')
        res_ts = minimize(
            chi2_ts, x0=[self.h_planck_measured, 1.0, 5.0], bounds=self.bounds_ts_cmb, method='L-BFGS-B')

        self.results['CMB'] = {
            'LCDM': {
                'params': res_lcdm.x,
                'chi2': res_lcdm.fun,
                'AIC': res_lcdm.fun + 2 * 1,
                'BIC': res_lcdm.fun + 1 * np.log(len(z_vals)),
            },
            'Time-Scaling': {
                'params': res_ts.x,
                'chi2': res_ts.fun,
                'AIC': res_ts.fun + 2 * 2,
                'BIC': res_ts.fun + 2 * np.log(len(z_vals)),
            },
            'mu_data': {
                'z': z_vals,
                'mu': mu_cmb,
                'mu_err': mu_err
            }
        }

    def _fit_all(self):
        """
        Fit models to all datasets simultaneously (combined analysis)
        """
        print("Fitting combined analysis to all datasets...")

        def chi2_lcdm_combined(params):
            """Combined chi-squared for LCDM across all datasets"""
            H0, Om = params
            chi2_total = 0

            # Supernova contribution
            mu_model_sn = self._distance_modulus_LCDM(
                self.sn_data['z'], H0, Om, self.sn_M_B)
            chi2_total += np.sum(((self.sn_data['mu'] -
                                   mu_model_sn) / self.sn_data['mu_err'])**2)

            # GRB contribution
            mu_model_grb = self._distance_modulus_LCDM(
                self.grb_data['z'], H0, Om, self.grb_M_B)
            chi2_total += np.sum(((self.grb_data['mu'] -
                                   mu_model_grb) / self.grb_data['sigma_mu'])**2)

            # Cosmic Chronometers contribution
            H_model_cc = self._H_LCDM(self.cc_data['z'], H0, Om)
            chi2_total += np.sum(((self.cc_data['H'] -
                                   H_model_cc) / self.cc_data['H_err'])**2)

            # CMB contribution (simplified)
            if 'mu_data' in self.results.get('CMB', {}):
                z_cmb = self.results['CMB']['mu_data']['z']
                mu_cmb = self.results['CMB']['mu_data']['mu']
                mu_err_cmb = self.results['CMB']['mu_data']['mu_err']

                # Use the same CMB model as in _fit_cmb but with fixed mu0
                mu0_fixed = 5.0  # Or use the best-fit value from individual CMB fit
                mu_model_cmb = np.log10(
                    1 + z_cmb) + np.log10(H0 / self.h_planck_measured) + mu0_fixed
                chi2_total += np.sum(((mu_cmb - mu_model_cmb) / mu_err_cmb)**2)

            return chi2_total

        def chi2_time_scaling_combined(params):
            """Combined chi-squared for Time-Scaling model across all datasets"""
            H0, alpha = params
            chi2_total = 0

            # Supernova contribution
            mu_model_sn = self._distance_modulus_time_scaling(
                self.sn_data['z'], H0, alpha, self.sn_M_B)
            chi2_total += np.sum(((self.sn_data['mu'] -
                                   mu_model_sn) / self.sn_data['mu_err'])**2)

            # GRB contribution
            mu_model_grb = self._distance_modulus_time_scaling(
                self.grb_data['z'], H0, alpha, self.grb_M_B)
            chi2_total += np.sum(((self.grb_data['mu'] -
                                   mu_model_grb) / self.grb_data['sigma_mu'])**2)

            # Cosmic Chronometers contribution
            H_model_cc = self._H_time_scaling(self.cc_data['z'], H0, alpha)
            chi2_total += np.sum(((self.cc_data['H'] -
                                   H_model_cc) / self.cc_data['H_err'])**2)

            # CMB contribution (simplified)
            if 'mu_data' in self.results.get('CMB', {}):
                z_cmb = self.results['CMB']['mu_data']['z']
                mu_cmb = self.results['CMB']['mu_data']['mu']
                mu_err_cmb = self.results['CMB']['mu_data']['mu_err']

                # Use the same CMB model as in _fit_cmb but with fixed mu0
                mu0_fixed = 5.0  # Or use the best-fit value from individual CMB fit
                mu_model_cmb = np.log10(
                    1e3 / (1 + z_cmb)**(1 - alpha) * (H0 / self.h_planck_measured)) + mu0_fixed
                chi2_total += np.sum(((mu_cmb - mu_model_cmb) / mu_err_cmb)**2)

            return chi2_total

        # Perform optimization
        res_lcdm_combined = differential_evolution(
            chi2_lcdm_combined, bounds=self.bounds_lcdm)
        res_ts_combined = differential_evolution(
            chi2_time_scaling_combined, bounds=self.bounds_ts)

        # Calculate total number of data points
        n_data_total = (len(self.sn_data['z']) + len(self.grb_data) +
                        len(self.cc_data['z']))

        # Add CMB data points if available
        if 'mu_data' in self.results.get('CMB', {}):
            n_data_total += len(self.results['CMB']['mu_data']['z'])

        # Store results
        self.results['COMBINED'] = {
            'LCDM': {
                'params': res_lcdm_combined.x,
                'chi2': res_lcdm_combined.fun,
                'AIC': res_lcdm_combined.fun + 2*2,  # 2 parameters
                'BIC': res_lcdm_combined.fun + 2*np.log(n_data_total),
                'chi2_reduced': res_lcdm_combined.fun / (n_data_total - 2)
            },
            'Time-Scaling': {
                'params': res_ts_combined.x,
                'chi2': res_ts_combined.fun,
                'AIC': res_ts_combined.fun + 2*2,  # 2 parameters
                'BIC': res_ts_combined.fun + 2*np.log(n_data_total),
                'chi2_reduced': res_ts_combined.fun / (n_data_total - 2)
            }
        }

        print(
            f"Combined analysis completed with {n_data_total} total data points")

    def _fit_all_unified(self):
        """
        Fit models to all datasets simultaneously with unified M_B treatment
        """
        print("Fitting unified analysis to all datasets...")

        def chi2_lcdm_unified(params):
            """Unified chi-squared for LCDM with simultaneous M_B fitting"""
            H0, Om, M_B_sn, M_B_grb = params  # Fit for both M_B values
            chi2_total = 0

            # Supernova contribution
            mu_model_sn = self._distance_modulus_LCDM(
                self.sn_data['z'], H0, Om, M_B_sn)
            chi2_total += np.sum(((self.sn_data['mu'] -
                                   mu_model_sn) / self.sn_data['mu_err'])**2)

            # GRB contribution
            mu_model_grb = self._distance_modulus_LCDM(
                self.grb_data['z'], H0, Om, M_B_grb)
            chi2_total += np.sum(((self.grb_data['mu'] -
                                   mu_model_grb) / self.grb_data['sigma_mu'])**2)

            # Cosmic Chronometers contribution (no M_B dependence)
            H_model_cc = self._H_LCDM(self.cc_data['z'], H0, Om)
            chi2_total += np.sum(((self.cc_data['H'] -
                                   H_model_cc) / self.cc_data['H_err'])**2)

            # CMB contribution (no M_B dependence)
            if 'mu_data' in self.results.get('CMB', {}):
                z_cmb = self.results['CMB']['mu_data']['z']
                mu_cmb = self.results['CMB']['mu_data']['mu']
                mu_err_cmb = self.results['CMB']['mu_data']['mu_err']

                mu0_fixed = 5.0
                mu_model_cmb = np.log10(
                    1 + z_cmb) + np.log10(H0 / self.h_planck_measured) + mu0_fixed
                chi2_total += np.sum(((mu_cmb - mu_model_cmb) / mu_err_cmb)**2)

            return chi2_total

        def chi2_time_scaling_unified(params):
            """Unified chi-squared for Time-Scaling model with simultaneous M_B fitting"""
            H0, alpha, M_B_sn, M_B_grb = params
            chi2_total = 0

            # Supernova contribution
            mu_model_sn = self._distance_modulus_time_scaling(
                self.sn_data['z'], H0, alpha, M_B_sn)
            chi2_total += np.sum(((self.sn_data['mu'] -
                                   mu_model_sn) / self.sn_data['mu_err'])**2)

            # GRB contribution
            mu_model_grb = self._distance_modulus_time_scaling(
                self.grb_data['z'], H0, alpha, M_B_grb)
            chi2_total += np.sum(((self.grb_data['mu'] -
                                   mu_model_grb) / self.grb_data['sigma_mu'])**2)

            # Cosmic Chronometers contribution
            H_model_cc = self._H_time_scaling(self.cc_data['z'], H0, alpha)
            chi2_total += np.sum(((self.cc_data['H'] -
                                   H_model_cc) / self.cc_data['H_err'])**2)

            # CMB contribution
            if 'mu_data' in self.results.get('CMB', {}):
                z_cmb = self.results['CMB']['mu_data']['z']
                mu_cmb = self.results['CMB']['mu_data']['mu']
                mu_err_cmb = self.results['CMB']['mu_data']['mu_err']

                mu0_fixed = 5.0
                mu_model_cmb = np.log10(
                    1e3 / (1 + z_cmb)**(1 - alpha) * (H0 / self.h_planck_measured)) + mu0_fixed
                chi2_total += np.sum(((mu_cmb - mu_model_cmb) / mu_err_cmb)**2)

            return chi2_total

        def chi2_lcdm_cc_only(params):
            """LCDM chi-squared for CC data only"""
            H0, Om = params
            H_model_cc = self._H_LCDM(self.cc_data['z'], H0, Om)
            return np.sum(((self.cc_data['H'] - H_model_cc) / self.cc_data['H_err'])**2)

        def chi2_time_scaling_cc_only(params):
            """Time-Scaling chi-squared for CC data only"""
            H0, alpha = params
            H_model_cc = self._H_time_scaling(self.cc_data['z'], H0, alpha)
            return np.sum(((self.cc_data['H'] - H_model_cc) / self.cc_data['H_err'])**2)

        # Bounds including M_B parameters
        bounds_lcdm_unified = [
            self.bounds_lcdm[0],  # H0 bounds
            self.bounds_lcdm[1],  # Om bounds
            (-25, 25),           # M_B_sn bounds (typical range)
            (-35, 25)            # M_B_grb bounds (typical range)
        ]

        bounds_ts_unified = [
            self.bounds_ts[0],    # H0 bounds
            self.bounds_ts[1],    # alpha bounds
            (-25, 25),           # M_B_sn bounds
            (-35, 25)            # M_B_grb bounds
        ]

        # Perform optimization for unified fits
        res_lcdm_unified = differential_evolution(
            chi2_lcdm_unified, bounds=bounds_lcdm_unified)
        res_ts_unified = differential_evolution(
            chi2_time_scaling_unified, bounds=bounds_ts_unified)

        # Perform optimization for CC-only fits
        res_lcdm_cc = differential_evolution(
            chi2_lcdm_cc_only, bounds=self.bounds_lcdm)
        res_ts_cc = differential_evolution(
            chi2_time_scaling_cc_only, bounds=self.bounds_ts)

        # Calculate total data points
        n_data_total = (len(self.sn_data['z']) + len(self.grb_data) +
                        len(self.cc_data['z']))
        if 'mu_data' in self.results.get('CMB', {}):
            n_data_total += len(self.results['CMB']['mu_data']['z'])

        n_data_cc = len(self.cc_data['z'])

        # Store results with parameter names
        self.results['COMBINED'] = {
            'LCDM': {
                'H0': res_lcdm_unified.x[0],
                'Om': res_lcdm_unified.x[1],
                'M_B_sn': res_lcdm_unified.x[2],
                'M_B_grb': res_lcdm_unified.x[3],
                'params': res_lcdm_unified.x,
                'chi2': res_lcdm_unified.fun,
                'AIC': res_lcdm_unified.fun + 2*4,  # 4 parameters now
                'BIC': res_lcdm_unified.fun + 4*np.log(n_data_total),
                'chi2_reduced': res_lcdm_unified.fun / (n_data_total - 4)
            },
            'Time-Scaling': {
                'H0': res_ts_unified.x[0],
                'alpha': res_ts_unified.x[1],
                'M_B_sn': res_ts_unified.x[2],
                'M_B_grb': res_ts_unified.x[3],
                'params': res_ts_unified.x,
                'chi2': res_ts_unified.fun,
                'AIC': res_ts_unified.fun + 2*4,  # 4 parameters now
                'BIC': res_ts_unified.fun + 4*np.log(n_data_total),
                'chi2_reduced': res_ts_unified.fun / (n_data_total - 4)
            }
        }

        # Store CC-only results
        self.results['CC'] = {
            'LCDM': {
                'H0': res_lcdm_cc.x[0],
                'Om': res_lcdm_cc.x[1],
                'params': res_lcdm_cc.x,
                'chi2': res_lcdm_cc.fun,
                'AIC': res_lcdm_cc.fun + 2*2,
                'BIC': res_lcdm_cc.fun + 2*np.log(n_data_cc),
                'chi2_reduced': res_lcdm_cc.fun / (n_data_cc - 2)
            },
            'Time-Scaling': {
                'H0': res_ts_cc.x[0],
                'alpha': res_ts_cc.x[1],
                'params': res_ts_cc.x,
                'chi2': res_ts_cc.fun,
                'AIC': res_ts_cc.fun + 2*2,
                'BIC': res_ts_cc.fun + 2*np.log(n_data_cc),
                'chi2_reduced': res_ts_cc.fun / (n_data_cc - 2)
            }
        }

    def print_results(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL FITTING RESULTS")
        print("="*80)

        datasets = ['CC', 'SN', 'GRB', 'CMB', 'COMBINED']
        models = ['LCDM', 'Time-Scaling']

        for dataset in datasets:
            if dataset in self.results:
                print(f"\n{dataset} Dataset:")
                print("-" * 40)

                for model in models:
                    if model in self.results[dataset]:
                        res = self.results[dataset][model]
                        params = res['params']

                        if model == 'LCDM':
                            if dataset == 'CMB':
                                print(f"{model:15s}: H₀={params[0]:.2f}")
                            else:
                                print(
                                    f"{model:15s}: H₀={params[0]:.2f}, Ωₘ={params[1]:.3f}")
                        else:  # Time-Scaling
                            print(
                                f"{model:15s}: H₀={params[0]:.2f}, α={params[1]:.3f}")

                        print(
                            f"{'':15s}  χ²={res['chi2']:.2f}, AIC={res['AIC']:.2f}, BIC={res['BIC']:.2f}")
                        if 'chi2_reduced' in res:
                            print(f"{'':15s}  χ²ᵣₑd={res['chi2_reduced']:.3f}")

                # Add comparison for combined results
                if dataset == 'COMBINED':
                    print("\n" + "-" * 40)
                    print("MODEL COMPARISON (Combined Analysis):")
                    lcdm_aic = self.results['COMBINED']['LCDM']['AIC']
                    ts_aic = self.results['COMBINED']['Time-Scaling']['AIC']
                    delta_aic = abs(lcdm_aic - ts_aic)
                    better_model = "LCDM" if lcdm_aic < ts_aic else "Time-Scaling"
                    print(f"ΔAIC = {delta_aic:.2f} (favors {better_model})")

                    lcdm_bic = self.results['COMBINED']['LCDM']['BIC']
                    ts_bic = self.results['COMBINED']['Time-Scaling']['BIC']
                    delta_bic = abs(lcdm_bic - ts_bic)
                    better_model_bic = "LCDM" if lcdm_bic < ts_bic else "Time-Scaling"
                    print(
                        f"ΔBIC = {delta_bic:.2f} (favors {better_model_bic})")

    def plot_observations_fit(self):
        """Create comprehensive plot of observational data fits"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Cosmic Chronometers plot
        ax = axes[0, 0]
        z_cc = self.cc_data['z']
        H_obs = self.cc_data['H']
        H_err = self.cc_data['H_err']

        ax.errorbar(z_cc, H_obs, yerr=H_err, fmt='o', color='black',
                    label='CC Data', markersize=4)

        z_plot_cc = np.linspace(min(z_cc), max(z_cc), 100)
        lcdm_params = self.results['CC']['LCDM']['params']
        ts_params = self.results['CC']['Time-Scaling']['params']

        H_lcdm = self._H_LCDM(z_plot_cc, *lcdm_params)
        H_ts = self._H_time_scaling(z_plot_cc, *ts_params)

        ax.plot(z_plot_cc, H_lcdm, '-', color='red',
                label='ΛCDM', linewidth=2)
        ax.plot(z_plot_cc, H_ts, '--', color='blue',
                label='Time-Scaling', linewidth=2)

        ax.set_xlabel('Redshift z')
        ax.set_ylabel('H(z) [km/s/Mpc]')
        ax.set_title('Cosmic Chronometers')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add informative textbox for CC
        cc_lcdm_chi2 = self.results['CC']['LCDM'].get('chi2', 'N/A')
        cc_ts_chi2 = self.results['CC']['Time-Scaling'].get('chi2', 'N/A')
        cc_text = f'ΛCDM: H₀={lcdm_params[0]:.1f}'
        if cc_lcdm_chi2 != 'N/A':
            cc_text += f', χ²={cc_lcdm_chi2:.2f}'
        cc_text += f'\nTime-Scaling: H₀={ts_params[0]:.1f}, α={ts_params[1]:.3f}'
        if cc_ts_chi2 != 'N/A':
            cc_text += f', χ²={cc_ts_chi2:.2f}'
        ax.text(0.05, 0.05, cc_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Supernova plot
        ax = axes[0, 1]
        z_sn = self.sn_data['z']
        mu_obs = self.sn_data['mu']
        mu_err = self.sn_data['mu_err']

        ax.errorbar(z_sn, mu_obs, yerr=mu_err, fmt='o', color='black',
                    label='SN Data', alpha=0.7, markersize=2)

        z_plot = np.linspace(min(z_sn), max(z_sn), 100)
        lcdm_params = self.results['SN']['LCDM']['params']
        ts_params = self.results['SN']['Time-Scaling']['params']

        mu_lcdm = self._distance_modulus_LCDM(
            z_plot, *lcdm_params, M_B=self.sn_M_B)
        mu_ts = self._distance_modulus_time_scaling(
            z_plot, *ts_params, M_B=self.sn_M_B)

        ax.plot(z_plot, mu_lcdm, '-', color='red', label='ΛCDM', linewidth=2)
        ax.plot(z_plot, mu_ts, '--', color='blue',
                label='Time-Scaling', linewidth=2)

        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Distance Modulus μ')
        ax.set_title('Type Ia Supernovae')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add informative textbox for SN
        sn_lcdm_chi2 = self.results['SN']['LCDM'].get('chi2', 'N/A')
        sn_ts_chi2 = self.results['SN']['Time-Scaling'].get('chi2', 'N/A')
        sn_text = f'ΛCDM: H₀={lcdm_params[0]:.1f}, Ωₘ={lcdm_params[1]:.3f}'
        if sn_lcdm_chi2 != 'N/A':
            sn_text += f', χ²={sn_lcdm_chi2:.2f}'
        sn_text += f'\nTime-Scaling: H₀={ts_params[0]:.1f}, α={ts_params[1]:.3f}'
        if sn_ts_chi2 != 'N/A':
            sn_text += f', χ²={sn_ts_chi2:.2f}'
        ax.text(0.05, 0.05, sn_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # GRB plot
        ax = axes[1, 0]
        z_grb = self.grb_data['z']
        mu_grb = self.grb_data['mu']
        err_grb = self.grb_data['sigma_mu']

        ax.errorbar(z_grb, mu_grb, yerr=err_grb, fmt='o', color='black',
                    label='GRB Data', markersize=4)

        z_plot_grb = np.linspace(min(z_grb), max(z_grb), 100)
        lcdm_params_grb = self.results['GRB']['LCDM']['params']
        ts_params_grb = self.results['GRB']['Time-Scaling']['params']

        mu_lcdm_grb = self._distance_modulus_LCDM(
            z_plot_grb, *lcdm_params_grb, M_B=self.grb_M_B)
        mu_ts_grb = self._distance_modulus_time_scaling(
            z_plot_grb, *ts_params_grb, M_B=self.grb_M_B)

        ax.plot(z_plot_grb, mu_lcdm_grb, '-', color='red',
                label='ΛCDM', linewidth=2)
        ax.plot(z_plot_grb, mu_ts_grb, '--', color='blue',
                label='Time-Scaling', linewidth=2)

        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Distance Modulus μ')
        ax.set_title('Gamma-Ray Bursts')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add informative textbox for GRB
        grb_lcdm_chi2 = self.results['GRB']['LCDM'].get('chi2', 'N/A')
        grb_ts_chi2 = self.results['GRB']['Time-Scaling'].get('chi2', 'N/A')
        grb_text = f'ΛCDM: H₀={lcdm_params_grb[0]:.1f}, Ωₘ={lcdm_params_grb[1]:.3f}'
        if grb_lcdm_chi2 != 'N/A':
            grb_text += f', χ²={grb_lcdm_chi2:.2f}'
        grb_text += f'\nTime-Scaling: H₀={ts_params_grb[0]:.1f}, α={ts_params_grb[1]:.3f}'
        if grb_ts_chi2 != 'N/A':
            grb_text += f', χ²={grb_ts_chi2:.2f}'
        ax.text(0.05, 0.05, grb_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # CMB plot
        ax = axes[1, 1]

        # Real Planck-transformed data
        z_vals = self.results['CMB']['mu_data']['z']
        mu_vals = self.results['CMB']['mu_data']['mu']
        mu_err = self.results['CMB']['mu_data']['mu_err']

        ax.errorbar(z_vals, mu_vals, yerr=mu_err, fmt='.', color='black',
                    label='Planck-derived $\\mu_{\\mathrm{CMB}}$', alpha=0.5)

        # Smooth model fits
        z_dense = np.linspace(min(z_vals), max(z_vals), 500)

        # ΛCDM model parameters
        H0_lcdm, mu0_lcdm = self.results['CMB']['LCDM']['params']
        mu_lcdm = np.log10(1 + z_dense) + np.log10(H0_lcdm /
                                                   self.h_planck_measured) + mu0_lcdm

        # Time-Scaling model parameters
        H0_ts, alpha_ts, mu0_ts = self.results['CMB']['Time-Scaling']['params']
        mu_ts = np.log10(1e3 / (1 + z_dense) ** (1 - alpha_ts)
                         * (H0_ts / self.h_planck_measured)) + mu0_ts

        # Plot model curves
        ax.plot(z_dense, mu_lcdm, 'r-', linewidth=2,
                label=fr'ΛCDM Fit')
        ax.plot(z_dense, mu_ts, 'b--', linewidth=2,
                label=fr'Time-Scaling Fit')

        # Axis and legend
        ax.set_xlabel('Redshift z (ℓ / 14)')
        ax.set_ylabel(r'$\mu_{\mathrm{CMB}} = \log_{10} D_\ell$')
        ax.set_title('Planck 2018 TT')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Enhanced informative textbox for CMB (keeping the detailed format from original)
        cmb_lcdm_chi2 = self.results['CMB']['LCDM'].get('chi2', 'N/A')
        cmb_ts_chi2 = self.results['CMB']['Time-Scaling'].get('chi2', 'N/A')
        cmb_text = f'ΛCDM: H₀={H0_lcdm:.2f}, μ₀={mu0_lcdm:.2f}'
        if cmb_lcdm_chi2 != 'N/A':
            cmb_text += f', χ²={cmb_lcdm_chi2:.2f}'
        cmb_text += f'\nTime-Scaling: H₀={H0_ts:.2f}, α={alpha_ts:.2f}, μ₀={mu0_ts:.2f}'
        if cmb_ts_chi2 != 'N/A':
            cmb_text += f', χ²={cmb_ts_chi2:.2f}'

        ax.text(0.05, 0.05, cmb_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_observations_fit_combined(self):
        # COMBINED plot - Corrected version
        fig, ax = plt.subplots(figsize=(16, 12))

        # Plot all datasets together with different colors/markers
        # Supernova data
        ax.errorbar(self.sn_data['z'], self.sn_data['mu'], yerr=self.sn_data['mu_err'],
                    fmt='o', color='red', alpha=0.6, markersize=4, label='Supernovae')

        # GRB data
        ax.errorbar(self.grb_data['z'], self.grb_data['mu'], yerr=self.grb_data['sigma_mu'],
                    fmt='^', color='blue', alpha=0.6, markersize=4, label='GRBs')

        # CMB data (if available)
        if 'mu_data' in self.results.get('CMB', {}):
            z_cmb = self.results['CMB']['mu_data']['z']
            mu_cmb = self.results['CMB']['mu_data']['mu']
            mu_err_cmb = self.results['CMB']['mu_data']['mu_err']
            ax.errorbar(z_cmb, mu_cmb, yerr=mu_err_cmb, fmt='s', color='green',
                        alpha=0.6, markersize=4, label='CMB-derived')

        # Create redshift range for smooth model curves
        z_min = min(np.min(self.sn_data['z']), np.min(self.grb_data['z']))
        z_max = max(np.max(self.sn_data['z']), np.max(self.grb_data['z']))
        if 'mu_data' in self.results.get('CMB', {}):
            z_min = min(z_min, np.min(self.results['CMB']['mu_data']['z']))
            z_max = max(z_max, np.max(self.results['CMB']['mu_data']['z']))

        z_dense = np.linspace(z_min, z_max, 500)

        # ΛCDM model parameters (4 parameters: H0, Om, M_B_sn, M_B_grb)
        H0_lcdm, Om_lcdm, M_B_sn_lcdm, M_B_grb_lcdm = self.results['COMBINED']['LCDM']['params']

        # Time-Scaling model parameters (4 parameters: H0, alpha, M_B_sn, M_B_grb)
        H0_ts, alpha_ts, M_B_sn_ts, M_B_grb_ts = self.results['COMBINED']['Time-Scaling']['params']

        # Plot ΛCDM model curves for both SN and GRB using their respective M_B
        mu_lcdm_sn = self._distance_modulus_LCDM(
            z_dense, H0_lcdm, Om_lcdm, M_B_sn_lcdm)
        mu_lcdm_grb = self._distance_modulus_LCDM(
            z_dense, H0_lcdm, Om_lcdm, M_B_grb_lcdm)

        # Plot Time-Scaling model curves
        mu_ts_sn = self._distance_modulus_time_scaling(
            z_dense, H0_ts, alpha_ts, M_B_sn_ts)
        mu_ts_grb = self._distance_modulus_time_scaling(
            z_dense, H0_ts, alpha_ts, M_B_grb_ts)

        # Plot model curves
        ax.plot(z_dense, mu_lcdm_sn, 'r-',
                linewidth=2, label='ΛCDM (SN scale)')
        ax.plot(z_dense, mu_lcdm_grb, 'r:',
                linewidth=2, label='ΛCDM (GRB scale)')
        ax.plot(z_dense, mu_ts_sn, 'b-', linewidth=2,
                label='Time-Scaling (SN scale)')
        ax.plot(z_dense, mu_ts_grb, 'b:', linewidth=2,
                label='Time-Scaling (GRB scale)')

        plt.tight_layout()
        return fig

    def run_mcmc_analysis(self, n_walkers=64, n_steps=1000):
        """Run MCMC analysis for parameter estimation"""

        print("=== Running MCMC Analysis ===")

        def log_posterior_lcdm(params):
            H0, Om = params
            if not (60 < H0 < 80) or not (0.05 < Om < 0.60):
                return -np.inf

            # Combined likelihood
            chi2_total = 0
            mu_model = self._distance_modulus_LCDM(
                self.sn_data['z'], H0, Om, M_B=self.sn_M_B)
            chi2_total += np.sum(((self.sn_data['mu'] -
                                 mu_model) / self.sn_data['mu_err'])**2)

            dm_model = self._DM_over_rs_LCDM(
                self.grb_data['z'], H0, Om)
            chi2_total += np.sum(((self.grb_data['mu'] -
                                 dm_model) / self.grb_data['sigma_mu'])**2)

            H_model = self._H_LCDM(self.cc_data['z'], H0, Om)
            chi2_total += np.sum(((self.cc_data['H'] -
                                 H_model) / self.cc_data['H_err'])**2)

            return -0.5 * chi2_total

        def log_posterior_ts(params):
            H0, alpha = params
            if not (60 < H0 < 80) or not (0.5 < alpha < 3.0):
                return -np.inf

            chi2_total = 0
            mu_model = self._distance_modulus_time_scaling(
                self.sn_data['z'], H0, alpha, M_B=self.sn_M_B)
            chi2_total += np.sum(((self.sn_data['mu'] -
                                 mu_model) / self.sn_data['mu_err'])**2)

            dm_model = self._DM_over_rs_time_scaling(
                self.grb_data['z'], H0, alpha)
            chi2_total += np.sum(((self.grb_data['mu'] -
                                 dm_model) / self.grb_data['sigma_mu'])**2)

            H_model = self._H_time_scaling(self.cc_data['z'], H0, alpha)
            chi2_total += np.sum(((self.cc_data['H'] -
                                 H_model) / self.cc_data['H_err'])**2)

            return -0.5 * chi2_total

        # Run MCMC for both models
        mcmc_results = {}

        for model_name, log_post, init_params in [
            ('LCDM', log_posterior_lcdm, [70, 0.3]),
            ('Time-Scaling', log_posterior_ts, [70, 1.3])
        ]:
            print(f"Running MCMC for {model_name}...")

            ndim = len(init_params)
            pos = [init_params + 1e-4 *
                   np.random.randn(ndim) for i in range(n_walkers)]

            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_post)
            sampler.run_mcmc(pos, n_steps, progress=True)

            # Remove burn-in
            samples = sampler.get_chain(discard=n_steps//4, flat=True)

            mcmc_results[model_name] = {
                'samples': samples,
                'mean': np.mean(samples, axis=0),
                'std': np.std(samples, axis=0)
            }

            print(f"{model_name} MCMC results:")
            if model_name == 'LCDM':
                print(
                    f"  H0 = {mcmc_results[model_name]['mean'][0]:.2f} ± {mcmc_results[model_name]['std'][0]:.2f}")
                print(
                    f"  Ωm = {mcmc_results[model_name]['mean'][1]:.2f} ± {mcmc_results[model_name]['std'][1]:.2f}")
            else:
                print(
                    f"  H0 = {mcmc_results[model_name]['mean'][0]:.2f} ± {mcmc_results[model_name]['std'][0]:.2f}")
                print(
                    f"  α = {mcmc_results[model_name]['mean'][1]:.2f} ± {mcmc_results[model_name]['std'][1]:.2f}")

        return mcmc_results

    def fit_models(self):
        print("=== Fitting Models to All Observational Data ===")

        # Fit to existing datasets
        self._fit_supernovae()
        self._fit_grb()
        self._fit_cosmic_chronometers()
        self._fit_cmb()
        self._fit_all_unified()

        self.print_results()


class ScalarFieldDynamics:
    """
    Handles scalar field evolution and time asymmetry analysis
    as described in the scalar-tensor gravity framework
    """

    def __init__(self, xi: float):
        self.c = 299792.458  # km/s
        self.G = 6.67430e-11  # SI units
        self.xi = xi  # Positive non-minimal coupling parameter

    def quadratic_potential(self, alpha, m=1.0):
        """V(α) = (1/2)m²α²"""
        return 0.5 * m**2 * alpha**2

    def quadratic_potential_derivative(self, alpha, m=1.0):
        """dV/dα for quadratic potential"""
        return m**2 * alpha

    def cosine_potential(self, alpha, V0=1.0):
        """V(α) = V₀cos(α)"""
        return V0 * np.cos(alpha)

    def cosine_potential_derivative(self, alpha, V0=1.0):
        """dV/dα for cosine potential"""
        return -V0 * np.sin(alpha)

    def asymmetric_potential(self, alpha):
        """V(α) = α³sin(α) - asymmetric potential"""
        return alpha**3 * np.sin(alpha)

    def asymmetric_potential_derivative(self, alpha):
        """dV/dα for asymmetric potential"""
        return 3 * alpha**2 * np.sin(alpha) + alpha**3 * np.cos(alpha)

    def field_equation(self, t, y, potential_type='asymmetric', **kwargs):
        """
        Scalar field equation of motion: α̈ + 3Hα̇ + ξRα + dV/dα = 0
        where y = [α, α̇]
        """
        alpha, alpha_dot = y

        # Hubble parameter H ≈ α/t (simplified)
        H = alpha / t if t > 0 else 1e-10

        # Ricci scalar R ≈ 12H² (quasi-de Sitter approximation)
        R = 12 * H**2

        # Choose potential and its derivative
        if potential_type == 'quadratic':
            dV_dalpha = self.quadratic_potential_derivative(alpha, **kwargs)
        elif potential_type == 'cosine':
            dV_dalpha = self.cosine_potential_derivative(alpha, **kwargs)
        elif potential_type == 'asymmetric':
            dV_dalpha = self.asymmetric_potential_derivative(alpha)
        else:
            dV_dalpha = 0

        # Field equation: α̈ = -3Hα̇ - ξRα - dV/dα
        alpha_2dot = -3 * H * alpha_dot - self.xi * R * alpha - dV_dalpha

        return [alpha_dot, alpha_2dot]

    def evolve_field(self, t_span, initial_conditions, potential_type='asymmetric',
                     direction='forward', n_points=1000, **kwargs):
        """
        Evolve scalar field forward or backward in time
        """
        if direction == 'backward':
            t_span = [-t_span[1], -t_span[0]]

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def field_eq_wrapper(t, y):
            # Handle time direction
            actual_t = abs(t) if abs(t) > 1e-10 else 1e-10
            return self.field_equation(actual_t, y, potential_type, **kwargs)

        try:
            sol = solve_ivp(field_eq_wrapper, t_span, initial_conditions,
                            t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)

            if direction == 'backward':
                return -sol.t[::-1], sol.y[:, ::-1]
            else:
                return sol.t, sol.y

        except Exception as e:
            print(f"Integration failed: {e}")
            return None, None

    def plot_log_divergence(self, asymmetry_results):
        """
        Plot log-divergence curves with linear fit used for Lyapunov exponent.
        Helps verify sign and slope quality.
        """
        potentials = ['quadratic', 'cosine', 'asymmetric']
        directions = ['forward', 'backward']
        fig, axes = plt.subplots(len(potentials), 2, figsize=(
            14, 8), sharex=True, sharey=True)
        fig.suptitle("Lyapunov: log(Δα / Δα₀) with Fit", fontsize=16)

        for i, pot in enumerate(potentials):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                res = asymmetry_results[pot][direction]
                t = res['t']
                log_div = res.get('log_div', None)
                lyap = res.get('lyapunov', float('nan'))

                if log_div is not None and np.any(np.isfinite(log_div)):
                    # Fit line to valid region
                    valid = np.isfinite(log_div) & (log_div > -1e5)
                    if np.sum(valid) > 3:
                        coeffs = np.polyfit(t[valid], log_div[valid], 1)
                        fit_line = np.polyval(coeffs, t)

                        ax.plot(t, log_div, label='log(Δα)', color='crimson')
                        ax.plot(t, fit_line, '--',
                                label=f'Fit: λ ≈ {coeffs[0]:.3f}', color='black')
                    else:
                        ax.plot(t, log_div, label='log(Δα)',
                                color='gray', linestyle='dotted')
                        ax.text(0.5, 0.5, "Insufficient valid data",
                                transform=ax.transAxes)

                    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
                    ax.set_title(
                        f"{pot.capitalize()} — {direction.capitalize()}")
                    ax.set_xlabel("Time t")
                    ax.set_ylabel("log(Δα / Δα₀)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def analyze_time_asymmetry(self, t_span=(0.1, 1e3), initial_conditions=[1.0, 0.1], delta=1e-6):
        """
        Analyze time asymmetry, directional bias, and Lyapunov divergence for various potentials.
        """
        potentials = ['quadratic', 'cosine', 'asymmetric']
        directions = ['forward', 'backward']
        results = {}

        def sine_fit(t, A, w, phi, offset):
            return A * np.sin(w * t + phi) + offset

        def compute_lyapunov(t, y_main, y_pert, delta):
            diff = np.abs(y_main - y_pert)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_div = np.log(diff / delta)
            valid = np.isfinite(log_div) & (log_div > -1e5)
            if np.any(valid):
                lyap = np.polyfit(t[valid], log_div[valid], 1)[0]
            else:
                lyap = float('nan')
            return lyap, log_div

        for pot_type in potentials:
            results[pot_type] = {}

            for direction in directions:
                t, y_main = self.evolve_field(
                    t_span, initial_conditions, pot_type, direction)
                _, y_pert = self.evolve_field(t_span, [initial_conditions[0] + delta, initial_conditions[1]],
                                              pot_type, direction)

                alpha = y_main[0] if y_main is not None else None
                alpha_dot = y_main[1] if y_main is not None else None

                # Try sine fit to alpha(t)
                if alpha is not None:
                    try:
                        peaks, _ = find_peaks(alpha)
                        period_est = np.mean(np.diff(t[peaks])) if len(
                            peaks) > 1 else None
                        freq_guess = 2 * np.pi / period_est if period_est else 1.0
                        params, _ = curve_fit(sine_fit, t, alpha, p0=[
                            0.5, freq_guess, 0, 0])
                    except Exception:
                        params = [np.nan] * 4
                else:
                    params = [np.nan] * 4

                # Lyapunov estimate
                lyap, log_div = compute_lyapunov(
                    t, y_main[0], y_pert[0], delta) if y_pert is not None else (float('nan'), None)

                results[pot_type][direction] = {
                    't': t,
                    'alpha': alpha,
                    'alpha_dot': alpha_dot,
                    'lyapunov': lyap,
                    'log_div': log_div,
                    'sine_fit': {
                        'amplitude': params[0],
                        'frequency': params[1],
                        'phase': params[2],
                        'offset': params[3]
                    }
                }

            # Compute directional bias and asymmetry std between forward and backward
            alpha_fwd = results[pot_type]['forward']['alpha']
            alpha_bwd = results[pot_type]['backward']['alpha']
            if alpha_fwd is not None and alpha_bwd is not None:
                if len(alpha_bwd) < len(alpha_fwd):
                    alpha_bwd = np.pad(alpha_bwd, (0, 1), constant_values=0)
                diff = alpha_fwd - alpha_bwd
                asym_std = np.std(diff)
                bias = np.mean(diff)
            else:
                asym_std = bias = float('nan')

            results[pot_type]['asymmetry'] = {
                'std': asym_std,
                'bias': bias
            }

        return results

    def plot_scalar_dynamics(self, asymmetry_results, downsample=3):
        """
        Plot scalar field evolution for different potentials with annotations.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        potentials = ['quadratic', 'cosine', 'asymmetric']
        titles = ['Quadratic V(α) = ½m²α²', 'Cosine V(α) = V₀cos(α)',
                  'Asymmetric V(α) = α³sin(α)']

        for i, (pot_type, title) in enumerate(zip(potentials, titles)):
            ax = axes[i]
            result = asymmetry_results[pot_type]

            # Downsampling for performance/clarity (optional)
            fwd_t = result['forward']['t'][::downsample]
            fwd_a = result['forward']['alpha'][::downsample]
            bwd_t = result['backward']['t'][::downsample]
            bwd_a = result['backward']['alpha'][::downsample]

            # Forward
            if fwd_t is not None:
                ax.plot(fwd_t, fwd_a, label='Forward',
                        color='royalblue', linewidth=0.8)

            # Backward
            if bwd_t is not None:
                ax.plot(bwd_t, bwd_a, label='Backward', linestyle='--',
                        color='crimson', linewidth=0.6)

            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Time t', fontsize=9)
            ax.set_ylabel('Field α(t)', fontsize=9)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True)

            # Annotations
            asymmetry_data = result.get('asymmetry', {})
            asymmetry_val = asymmetry_data.get('std', float('nan'))
            bias_val = asymmetry_data.get('bias', float('nan'))
            lyap_val_f = result['forward'].get('lyapunov', float('nan'))
            lyap_val_b = result['backward'].get('lyapunov', float('nan'))

            ax.text(0.02, 0.95,
                    f'Asymmetry: {asymmetry_val:.5f}\nBias: {bias_val:.5f}\nLyapunov (forward): {lyap_val_f:.5f}\nLyapunov (backward): {lyap_val_b:.5f}',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='linen'))

        plt.tight_layout()
        return fig


class CosmologicalExperiment:
    """
    Main experiment runner that coordinates all analyses
    """

    def __init__(self, xi: float = 1e-6):
        self.framework = CosmologicalFramework()
        self.data_analyzer = ObservationalDataAnalyzer()
        self.scalar_dynamics = ScalarFieldDynamics(xi)

    def run_complete_analysis(self, run_mcmc, mcmc_walkers=32, mcmc_steps=256):
        """
        Run the complete end-to-end analysis pipeline
        """
        print("="*60)
        print("COSMOLOGICAL TIME-SCALING MODEL: COMPLETE ANALYSIS")
        print("="*60)

        # Step 1: Display symbolic framework
        print("\nStep 1: Symbolic Framework")
        print("-"*30)
        self.framework.print_equations()

        # Step 2: Generate and fit observational data
        print("\nStep 2: Observational Data Analysis")
        print("-"*30)

        self.data_analyzer.load_real_data()
        self.data_analyzer.fit_models()

        # Create observational plots
        obs_fig = self.data_analyzer.plot_observations_fit()
        obs_combined_fig = self.data_analyzer.plot_observations_fit_combined()

        # Step 3: MCMC Analysis (optional)
        mcmc_results = None
        if run_mcmc:
            print(
                f"\nStep 3: MCMC Analysis ({mcmc_steps} steps, {mcmc_walkers} walkers)")
            print("-"*30)
            mcmc_results = self.data_analyzer.run_mcmc_analysis(n_walkers=mcmc_walkers,
                                                                n_steps=mcmc_steps)
            mcmc_fig = self.plot_corner_mcmc(mcmc_results)

        # Step 4: Scalar field dynamics
        print("\nStep 4: Scalar Field Dynamics & Time Asymmetry")
        print("-"*30)
        asymmetry_results = self.scalar_dynamics.analyze_time_asymmetry()
        dynamics_fig = self.scalar_dynamics.plot_scalar_dynamics(
            asymmetry_results)
        lyap_fig = self.scalar_dynamics.plot_log_divergence(asymmetry_results)

        # Save figures
        obs_fig.savefig('fig_obs.png')
        obs_combined_fig.savefig('fig_obs_combined.png')
        dynamics_fig.savefig('fig_dynamics.png')
        lyap_fig.savefig('fig_lyap.png')
        if run_mcmc:
            mcmc_fig.savefig('fig_mcmc.png')

        return {
            'observational_results': self.data_analyzer.results,
            'mcmc_results': mcmc_results,
            'asymmetry_results': asymmetry_results,
            'figures': {
                'observations': obs_fig,
                'observations_combined': obs_combined_fig,
                'mcmc': mcmc_fig if run_mcmc else None,
                'dynamics': dynamics_fig
            }
        }

    def plot_corner_mcmc(self, mcmc_results):
        """Create corner plots for MCMC results"""
        if mcmc_results is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for i, (model_name, results) in enumerate(mcmc_results.items()):
            if 'samples' in results:
                samples = results['samples']

                if model_name == 'LCDM':
                    labels = ['H₀', 'Ωₘ']
                    row = 0
                else:
                    labels = ['H₀', 'α']
                    row = 1

                # 2D posterior
                axes[row, 0].scatter(
                    samples[:, 0], samples[:, 1], alpha=0.4, s=1)
                axes[row, 0].set_xlabel(labels[0])
                axes[row, 0].set_ylabel(labels[1])
                axes[row, 0].set_title(f'{model_name} Joint Posterior')

                # 1D marginals
                axes[row, 1].hist(samples[:, 1], bins=50,
                                  alpha=0.7, density=True)
                axes[row, 1].set_xlabel(labels[1])
                axes[row, 1].set_title(f'{labels[1]} Marginal')

        plt.tight_layout()
        return fig


# Main execution function
def run_full_experiment():
    # Create and run experiment
    experiment = CosmologicalExperiment(xi=1e-4)
    results = experiment.run_complete_analysis(
        run_mcmc=True, mcmc_walkers=64, mcmc_steps=50)
    return results


if __name__ == "__main__":
    # Run the complete analysis
    results = run_full_experiment()
