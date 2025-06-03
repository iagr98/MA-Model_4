import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 5
N_x = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
var = 'N_x'                                                             # Update


def parallel_simulation(params):
    N_x = params                                                        # Update parameters
    print(f"Start simulation with {var}={N_x}")                         # Update parameter in second {}
    try:
        Sim = run_sim(N_x=N_x)                                          # Update inputs
        return {f"{var}": N_x,                                          # Update parameter in second place
                'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 
                'status': 'success'}    
    except Exception as e:
        print(f"Simulation failed by {var}={N_x}: {str(e)}")            # Update parameter in second {}
        return {f"{var}": N_x, 'error': str(e), 'status': 'failed'}     # Update parameter in second place

if __name__ == "__main__":
    parameters = [N_x_value for N_x_value in N_x]                       # Update parameter var_value, var_value & var 
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_N_D.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

    # Plot results
    df = pd.read_csv("simulation_results_parallel_N_D.csv")
    df.columns = df.columns.str.strip()
    plt.figure(figsize=(8, 5))
    plt.plot(df['N_x'], df['V_dis_total'], marker='o')                # Update parameter in first place
    # plt.xscale('log')  # da atol logarithmisch skaliert ist
    # plt.yscale('log')  # da atol logarithmisch skaliert ist
    plt.xlabel('N_x')                                                   # Change x-label
    plt.ylabel('V_dis')                                             # Change output variable if needed
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df['N_x'], df['Sep. Eff.'], marker='o')                    # Update parameter in first place
    plt.ylabel('Sep. Eff.')                                             # Change output variable if needed
    plt.xlabel('N_x')                                                   # Change x-label
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()