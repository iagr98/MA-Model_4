import sim_model as sim
import sim_parameters as sp
import numpy as np


def init_sim(filename, N_x=101):
    if (filename == "Paraffin_flut_20C.xlsx"):
        Set = sp.Settings(N_x=N_x, L=0.56, D=0.15, h_c_0=0.055, h_dis_0=0.04)
    elif(filename == "niba_V1.xlsx" or filename == "niba_V2.xlsx" or filename == "niba_V3.xlsx" or filename == "niba_V4.xlsx"):
        Set = sp.Settings(N_x=N_x, L=1.0, D=0.2, h_c_0=0.1, h_dis_0=0.03)
    else:
        print('Test does not belong to either Ye or Niba.')
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    return sim.input_simulation(Set, SubSys)

def run_sim(filename, N_D=15, N_x=101, a_tol=1e-6):
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