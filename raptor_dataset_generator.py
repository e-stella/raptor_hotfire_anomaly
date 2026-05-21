"""
============================================================
RAPTOR ENGINE SENSOR DATA - ANOMALY DETECTION PRACTICE
============================================================
Context:
    You are a software engineer on the Raptor Systems Modeling team.
    A hot-fire test was conducted on Raptor engine #R3-047.
    The test ran for 120 seconds at nominal throttle.
    
    You have been given a CSV of sensor readings sampled at 10 Hz.
    Something went wrong during the test. Your job is to find it.

Sensors recorded:
    - time_s          : time in seconds
    - chamber_pressure_bar : main combustion chamber pressure (bar)
    - fuel_flow_kgs   : fuel (methane) mass flow rate (kg/s)
    - ox_flow_kgs     : oxidizer (LOX) mass flow rate (kg/s)
    - turbine_temp_K  : fuel turbopump turbine inlet temperature (K)
    - thrust_kN       : measured thrust (kN)

Nominal operating conditions:
    - Chamber pressure: 350 bar ± 5 bar
    - Mixture ratio (ox/fuel): 3.6 ± 0.1
    - Turbine temp: 720 K ± 20 K
    - Thrust: 2100 kN ± 50 kN

============================================================
YOUR TASKS:
============================================================

TASK 1 — Load and inspect the data
    a) Load raptor_hotfire_R3047.csv into a pandas DataFrame
    b) Print shape, dtypes, and first 5 rows
    c) Check for any missing values

TASK 2 — Basic statistics
    a) Compute mean, std, min, max for all sensor channels
    b) Flag any channel whose mean falls outside nominal range
    c) Compute the mixture ratio (ox_flow / fuel_flow) over time
       and report its mean and std

TASK 3 — Time series visualization
    a) Plot all 5 sensor channels vs time on subplots
    b) Add horizontal lines showing nominal ± tolerance bands
    c) Visually identify the time window where anomaly occurs

TASK 4 — Anomaly detection
    a) Use a rolling window (window=50 samples = 5 seconds) to compute
       rolling mean and rolling std for chamber_pressure_bar
    b) Flag samples where the value deviates more than 3 sigma
       from the rolling mean
    c) Report: at what time does the anomaly start?
               how long does it last?
               what is the peak deviation?

TASK 5 — Root cause analysis
    a) During the anomaly window identified in Task 4:
       - Which OTHER sensors also show anomalous behavior?
       - Does mixture ratio change during this window?
       - Does turbine temp spike or drop?
    b) Based on the pattern across sensors, what is your hypothesis
       for root cause? (hint: think about what would cause chamber
       pressure to drop while turbine temp rises)

TASK 6 — Report
    Write a short function:
    
    def anomaly_report(df) -> dict:
        returns a dictionary with keys:
        {
            'anomaly_start_s': float,
            'anomaly_end_s': float,
            'anomaly_duration_s': float,
            'peak_pressure_deviation_bar': float,
            'affected_sensors': list,
            'hypothesis': str
        }

============================================================
DATASET GENERATOR — run this to create your CSV
============================================================
Run generate_dataset() first, then attempt the tasks above.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_dataset(seed=42, output_file='raptor_hotfire_R3047.csv'):
    """
    Generates a synthetic Raptor hot-fire test dataset with a hidden anomaly.
    Saves to CSV. Run this first before attempting the tasks.
    """
    np.random.seed(seed)
    
    dt = 0.1          # 10 Hz sampling
    t_end = 120.0     # 120 second test
    t = np.arange(0, t_end, dt)
    n = len(t)
    
    # --- Nominal values ---
    P_nom = 350.0      # bar
    mdot_f_nom = 78.0  # kg/s methane
    mdot_ox_nom = 280.8  # kg/s LOX  (MR = 3.6)
    T_turb_nom = 720.0  # K
    F_nom = 2100.0     # kN
    
    # --- Noise levels (sensor noise) ---
    noise_P = 1.2
    noise_f = 0.3
    noise_ox = 0.8
    noise_T = 3.0
    noise_F = 8.0
    
    # --- Nominal signals with noise ---
    chamber_pressure = P_nom + np.random.normal(0, noise_P, n)
    fuel_flow = mdot_f_nom + np.random.normal(0, noise_f, n)
    ox_flow = mdot_ox_nom + np.random.normal(0, noise_ox, n)
    turbine_temp = T_turb_nom + np.random.normal(0, noise_T, n)
    thrust = F_nom + np.random.normal(0, noise_F, n)
    
    # --- Inject anomaly: partial fuel valve restriction ---
    # Starts at t=67s, lasts 18 seconds
    # Effect: fuel flow drops -> mixture ratio rises -> 
    #         chamber pressure drops -> turbine temp spikes (fuel-rich preburner runs hotter)
    #         thrust drops proportionally
    
    anomaly_start = 67.0
    anomaly_end = 85.0
    anom_mask = (t >= anomaly_start) & (t < anomaly_end)
    
    # Fuel flow restriction ramps in over 2 seconds, holds, ramps out
    ramp_in = np.clip((t - anomaly_start) / 2.0, 0, 1)
    ramp_out = np.clip((anomaly_end - t) / 2.0, 0, 1)
    severity = np.where(anom_mask, np.minimum(ramp_in, ramp_out) * 0.85, 1.0)
    # severity = 1.0 nominal, drops to ~0.85 during anomaly (15% fuel restriction)
    
    fuel_flow_anom = fuel_flow * np.where(anom_mask, severity, 1.0)
    
    # Downstream effects
    MR_effect = mdot_ox_nom / (fuel_flow_anom)  # MR rises as fuel drops
    pressure_drop = np.where(anom_mask, (1 - severity) * 28.0, 0)  # up to ~22 bar drop
    chamber_pressure_anom = chamber_pressure - pressure_drop
    
    turbine_spike = np.where(anom_mask, (1 - severity) * 65.0, 0)  # temp rises
    turbine_temp_anom = turbine_temp + turbine_spike
    
    thrust_drop = np.where(anom_mask, (1 - severity) * 180.0, 0)
    thrust_anom = thrust - thrust_drop
    
    # --- Add one NaN to make it realistic (sensor dropout) ---
    drop_idx = np.random.randint(200, 400)
    turbine_temp_anom[drop_idx] = np.nan
    
    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        'time_s': np.round(t, 1),
        'chamber_pressure_bar': np.round(chamber_pressure_anom, 2),
        'fuel_flow_kgs': np.round(fuel_flow_anom, 3),
        'ox_flow_kgs': np.round(ox_flow, 3),
        'turbine_temp_K': np.round(turbine_temp_anom, 1),
        'thrust_kN': np.round(thrust_anom, 1)
    })
    
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    return df



if __name__ == '__main__':
    print("Generating dataset...")
    df = generate_dataset()
    print("\nDataset ready.")
