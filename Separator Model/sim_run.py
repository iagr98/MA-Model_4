import sim_model as sim
import sim_model_modified as sim_mod
import sim_parameters as sp
import constants
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
import os


def init_sim(filename, N_x=101):
    Set = sp.Settings(N_x)
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    return sim_mod.input_simulation(Set, SubSys)

def run_sim(filename, N_D=10, N_x=101, a_tol=1e-6):
    Sim = init_sim(filename, N_x)
    Sim.initial_conditions(N_D)
    Sim.simulate_ivp(atol=a_tol)
    return Sim


if __name__ == "__main__":

    filename = "Paraffin_flut_20C.xlsx"
    # filename = "niba_V2.xlsx"
    N_D = 15
    N_x = 101
    a_tol = 1e-6
    

    Sim = run_sim(filename, N_D=N_D, N_x=N_x, a_tol=a_tol)

    # Animationen

    # zu Phasenhöhe, Sauterdrchmesser und Hold-Up
    # plots = ['heights', 'phi_32', 'hold_up']
    # eSim.plot_anim(plots)

    # zu Tropfenanzahl
    # plots = ['N_j']
    # eSim.plot_anim(plots)

    # zu Tropfenanzahl + Phasenhöhe
    # plots = ['N_j','heights']
    # eSim.plot_anim(plots)

    # zu Phasenhöhe)
    plots = ['heights','phi_32', 'hold_up']
    Sim.plot_anim(plots)