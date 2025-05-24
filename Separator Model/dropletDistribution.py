def get_droplet_classes(d_32, path, d_max=3e-3, s=0.32, plot=False):
    """
    calculate log-normal DSD function from sauter mean diameter d32[m]
    input: d_32: sauter mean diameter of droplets in m
    input: d_max: maximum droplet diameter in m
    input: s: standard deviation of volume-based log-normal distribution (Kraume2004)
    input: plot: if True plot the distribution
    input: path: path to store results
    return n_count_rel: relative number-based probability of droplets for each class based on the derived volume-based log normal distribution
    return d_bins_center: bin center of droplet classes

    see Kraume, Influence of Physical Properties on Drop Size Distributions of Stirred Liquid-Liquid Dispersions, 2004
    and Ye, Effect of temperature on mixing and separation of stirred liquid/liquid dispersions over a wide range of dispersed phase fractions, 2023
    """
    import constants
    import numpy as np
    from scipy.stats import lognorm
    from scipy import stats
    import matplotlib.pyplot as plt

    # statistical representative number of droplets for volume distribution
    N_vol = int(1e6)
    # lognorm volume distribution of d/d_32
    dist = lognorm(s)

    # define bin edges (diameter class) equidistantly from numberClasses
    x = np.linspace(0, d_max / d_32, constants.N_D + 1)

    if plot == True:
        # plot lognorm distribution
        fig, ax = plt.subplots(1, 1)
        ax.plot(x * d_32 * 1e6, dist.pdf(x))
        ax.set_ylim([0, 1.2 * max(dist.pdf(x))])
        ax.set_xlim([0, d_max * 1e6])
        ax.set_xlabel("$d / \mathrm{\mu m}$")
        ax.set_ylabel("$q_3 / \mathrm{\mu m}^-1$")
        ax.set_title(
            "Volume-based probability density distribution \n $d_{32}$="
            + str(d_32 * 1e6)
            + "$\mu m$"
            + ", $d_{max}$="
            + str(d_max * 1e6)
            + "$\mu m$"
            + ", \n number of classes="
            + str(constants.N_D)
        )
        # save plot
        fig.savefig(path + "\lognorm_dist.png", dpi=1000)
        fig.savefig(path + "\lognorm_dist.eps", dpi=1000)
        fig.savefig(path + "\lognorm_dist.svg", dpi=1000)

    # divide sample points into bins hist[0] is the count and hist[1] the edges of bins
    hist = np.histogram(dist.rvs(N_vol, random_state=1), bins=x, density=False)
    # return middle value of bins boundary values
    d_bins = hist[1] * d_32
    d_bins_center = np.zeros(len(d_bins) - 1)
    for i in range(len(d_bins) - 1):
        d_bins_center[i] = (d_bins[i] + d_bins[i + 1]) / 2

    # transform volume based absolute distribution to number based relative distribution
    v_count_abs = hist[0]
    n_count_abs = np.zeros(len(v_count_abs))
    v_count_rel = np.zeros(len(v_count_abs))
    for i in range(len(v_count_abs)):
        n_count_abs[i] = v_count_abs[i] * 6 / (np.pi * d_bins_center[i] ** 3)
        v_count_rel[i] = v_count_abs[i] / sum(v_count_abs)
    # normalize number distribution
    n_count_rel = np.zeros(len(v_count_abs))
    for i in range(len(v_count_abs)):
        n_count_rel[i] = n_count_abs[i] / sum(n_count_abs)

    # optional plotting of transformed distribution
    if plot == True:
        fig, ax = plt.subplots(1, 1)
        ax.plot(d_bins_center * 1e6, v_count_rel, label="Volume-based")
        ax.plot(d_bins_center * 1e6, n_count_rel, label="Number-based")
        # ax.set_ylim([0,1])
        ax.set_xlim([0, d_max * 1e6])
        ax.set_xlabel("$d / \mathrm{\mu m}$")
        ax.set_ylabel("$h $")
        ax.set_xlim([0, d_max * 1e6])
        ax.set_ylim([0, 1.2 * max(np.append(v_count_rel, n_count_rel))])
        ax.set_title(
            "Relative distribution \n $d_{32}$="
            + str(d_32 * 1e6)
            + "$\mu m$"
            + ", $d_{max}$="
            + str(d_max * 1e6)
            + "$\mu m$"
            + ", \n number of classes="
            + str(constants.N_D)
        )
        ax.legend()
        # save plot
        fig.savefig(path + "\lognorm_dist_rel.png", dpi=1000)
        fig.savefig(path + "\lognorm_dist_rel.eps", dpi=1000)
        fig.savefig(path + "\lognorm_dist_rel.svg", dpi=1000)

        # plot histogram of number distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(d_bins_center * 1e6, n_count_rel, width=0.5 * d_bins_center[1] * 1e6)
        ax.set_xlabel("$d / \mathrm{\mu m}$")
        ax.set_ylabel("$h $")
        ax.set_title(
            "Relative number-based distribution \n $d_{32}$="
            + str(d_32 * 1e6)
            + "$\mu m$"
            + ", $d_{max}$="
            + str(d_max * 1e6)
            + "$\mu m$"
            + ", \n number of classes="
            + str(constants.N_D)
        )
        fig.savefig(path + "\lognorm_dist_rel_n.png", dpi=1000)
        fig.savefig(path + "\lognorm_dist_rel_n.eps", dpi=1000)
        fig.savefig(path + "\lognorm_dist_rel_n.svg", dpi=1000)
        # plot histogram of volume distribution
        # plot histogram of number distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(d_bins_center * 1e6, v_count_rel, width=0.5 * d_bins_center[1] * 1e6)
        ax.set_xlabel("$d / \mathrm{\mu m}$")
        ax.set_ylabel("$h $")
        ax.set_title(
            "Relative volume-based distribution \n $d_{32}$="
            + str(d_32 * 1e6)
            + "$\mu m$"
            + ", $d_{max}$="
            + str(d_max * 1e6)
            + "$\mu m$"
            + ", \n number of classes="
            + str(constants.N_D)
        )
        fig.savefig(path + "\lognorm_dist_rel_v.png", dpi=1000)
        fig.savefig(path + "\lognorm_dist_rel_v.eps", dpi=1000)
        fig.savefig(path + "\lognorm_dist_rel_v.svg", dpi=1000)

    return n_count_rel, d_bins_center


def get_V_d(D):
    # Volume of droplet with diameter m
    import numpy as np

    V_d = np.pi / 6 * D**3
    return V_d


def get_totalNumber_water_inlet(hold_up, d_32, d_max, V_mix, path):
    """
    calculates the total number of droplets entering the separator for a given hold up and volume of mixing that follows the volume-based lognormal distribution (Kraume2004)
    input: hold_up: hold up of org. in aq. phase entering the separator in 1
    input: d_32: sauter mean diameter of droplets in m
    input: d_max: maximum droplet diameter in m
    input: V_mix: Volume of mixer (volume of first aq. phase segment) in m3
    output: N_in_total: total number of droplets entering the separator in 1
    """
    # use minimize to calculate number of droplets
    from scipy import optimize
    import constants

    N_in_total = 1e4  # initial guess

    # relative number distribution at inlet
    n_count_rel, d_bins = get_droplet_classes(d_32, d_max=d_max, path=path)

    def f(N_in_total):

        # volume of dispered phase in m3
        V_disp = 0
        for i in range(constants.N_D):
            V_disp = V_disp + N_in_total * n_count_rel[i] * get_V_d(d_bins[i])
        # hold up of water in separator
        hold_up_calc = V_disp / V_mix
        return hold_up_calc - hold_up

    N_in_total = optimize.newton(f, N_in_total, rtol=1e-4)

    # converet number of droplets to integer
    N_in_total = int(N_in_total)
    # calculate hold up for found number of droplets
    hold_up_calc = f(N_in_total) + hold_up
    # print results
    # print hold_up_calc with 4 digits
    hold_up_calc = round(hold_up_calc, 4)
    # print("hold up: " + str(hold_up_calc))
    # print("number of droplets: " + str(N_in_total))
    return N_in_total, n_count_rel, d_bins


import numpy as np

# import physical properties
import Properties.properties_Butylacetate as prop

# number of segments
N_S = 200  # -
# number of droplet classes
N_D = 10  # -
EPS = 1e-12  # epsilon for numerical stability
POS_IN = np.zeros(N_D)  # position of droplet at inlet in vertical direction m

# Geometry data
R = 0.1  # radius of separator m
L = 1.8  # length of separator m
D_STIRRER = 0.1  # diameter of stirrer m
# parameter definition
G = 9.81  # gravity constant m/s^2

# property data from properties file
RHO_O = prop.RHO_O  # density of organic phase kg/m3
RHO_W = prop.RHO_W  # density of water phase kg/m3
ETA_O = prop.ETA_O  # viscosity of organic phase Pa*s
ETA_W = prop.ETA_W  # viscosity of water phase Pa*s
DELTA_RHO = prop.DELTA_RHO  # density difference in kg/m3
SIGMA = prop.SIGMA  # interfacial tension N/m
R_V = prop.R_V  # asymetric film drainage parameter

# R_IG = 8.314 # ideal gas constant J/mol*K
# RHO_G = 1.2 # density of gas kg/m3
# M_G = 28.97e-3 # molar mass of gas kg/mol

## Henschke
HA = 1e-20  # Hamaker constant J
EPSILON_DI = 1  # holdup at interface -
EPSILON_DP = 0.9  # holdup in dense-packed zone -
