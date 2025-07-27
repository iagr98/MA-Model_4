import numpy as np
from scipy.optimize import newton
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Funktion berechnet Höhe eines Kreissegments auf Basis des Kreisradius r und der Fläche A
def getHeight(A, r):
    eq = lambda h: A - r**2 * np.arccos(1 - h / r) + (r - h) * np.sqrt(2 * r * h - h**2)
    h0 = r / 2
    if A < 0:
        #print('Querschnitt kleiner Null: ' + str(A))
        return 0
    elif A > np.pi * r**2:
        #print('Querschnitt größer als zulässig: ' + str(A))
        return 2*r
    return newton(eq, h0)

def getHeightArray(A, r):
    h = np.zeros_like(A)
    for i in range(len(h)):
        h[i] = getHeight(A[i], r)
    return h

# Funktion berechnet die Fläche eines Kreissegments auf Basis des Kreisradiuses r und der Höhe h des Segments
def getArea(h, r):
    return r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(2 * r * h - h**2)

# Funktion berechnet dyn. Viskosität in der dicht gepackten Schicht nach Modell von Yaron und Gal-Or
def yaron(eta_c, eta_d, eps, eta_v=23e-3):
    al = eta_c / (eta_d + eta_v)
    ga = eps ** (1 / 3)
    omega = ((4 * ga ** 7 + 10 - (84 / 11) * ga ** 2 + 4 * al * (1 - ga ** 7)) /
             (10 * (1 - ga ** 10) - 25 * ga ** 3 * (1 - ga ** 4) + 10 * al * (1 - ga ** 3) * (1 - ga ** 7)))
    return eta_c * (1 + 5.5 * omega * eps)

# Funktion zum Glätten von Arrays
def smooth_array(arr, window_size=5):
                return np.convolve(arr, np.ones(window_size)/window_size, mode='same')

def calculate_volume_balance(Sim):
      """ Berechnet die Volumenbilanz für die Simulation
       Ausgabe in prozent """
      dV_ges = Sim.Sub.dV_ges
      _, u_d, u_c = Sim.velocities(Sim.V_dis[:,-1], Sim.V_d[:,-1], Sim.V_c[:,-1], Sim.N_j, Sim.Set.T, calc_balance=True)
      A_c = Sim.V_c[-1,-1] / Sim.Set.dl
      A_d = Sim.V_d[-1,-1] / Sim.Set.dl
      return 100*abs(dV_ges - u_c*A_c - u_d*A_d)/dV_ges

def calculate_cfl(Sim):
    u_dis, u_d, u_c = Sim.velocities(Sim.V_dis[:,-1], Sim.V_d[:,-1], Sim.V_c[:,-1], Sim.N_j, Sim.Set.T, calc_balance=True)
    u = max(u_dis, u_d, u_c)
    return u * Sim.Set.dt / Sim.Set.dl

# Input data
data = np.array([
    [0.05, 240, 180, 0.036],
    [0.05, 240, 200, 0.023],
    [0.05, 240, 250, 0.004],
    [0.05, 280, 180, 0.028],
    [0.05, 280, 200, 0.008],
    [0.05, 280, 250, 0.002],
    [0.1, 200, 180, 0.038],
    [0.1, 240, 180, 0.03],
    [0.1, 280, 180, 0.03],
    [0.2, 160, 180, 0.040],
    [0.2, 160, 200, 0.023],
    [0.2, 160, 250, 0.003],
    [0.2, 200, 180, 0.021],
    [0.2, 200, 200, 0.035],
    [0.2, 200, 250, 0.009],
    [0.2, 240, 180, -0.151],
    [0.2, 240, 200, 0.041],
    [0.2, 240, 250, 0.011],
    [0.2, 280, 180, -0.221],
    [0.2, 280, 200, 0.006],
    [0.2, 280, 250, 0.012],
    [0.3, 750, 205, 0.040],
    [0.3, 1000, 250, 6e-3],
    [0.3, 1250, 275, 0.0115],
    [0.4, 750, 225, 8.5e-3],
])

# Separate inputs and output
eps_0 = data[:, 0]
dV_ges = data[:, 1]
phi_0 = data[:, 2]
delta = data[:, 3]

# Stack inputs as points
points = np.column_stack((eps_0, dV_ges, phi_0))
interp_lin = LinearNDInterpolator(points, delta)
interp_nearest = NearestNDInterpolator(points, delta)


def E_interpolator(x):
    if x[2] > 300:
        raise ValueError("phi_0 exceeds maximum allowed value of 300 um")

    val = interp_lin(x)
    if np.isnan(val):
        return interp_nearest(x)  # Fallback extrapolation
    return val

# Example queries
# E_interpolator([0.2, 240, 205])