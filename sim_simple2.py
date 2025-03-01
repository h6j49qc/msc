import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define reaction rate constants (assumed reasonable values)
k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.06  # Inactivation rate of BCR-ABL (s⁻¹)
k_on1 = 1e2  # ATP binding rate (M⁻¹s⁻¹)
k_off1 = 0.1  # ATP unbinding rate (s⁻¹)
k_on3 = 1e2  # Imatinib binding rate (M⁻¹s⁻¹)
k_off3 = 0.00  # Imatinib unbinding rate (s⁻¹)
Kintake = 3e-10 # Imatinib intake Ms⁻¹ (dosage 600mg or approx. 1.2 millimole daily)

# Initial concentrations (M)
BcrAbl_active_0 = 5e-6  # 5 μM
BcrAbl_inactive_0 = 2e-6  # 2 μM
ATP_0 = 1.1e-3  # 1 mM (intracellular ATP concentration)
BcrAbl_ATP_0 = 0  # No bound ATP at start
Imatinib_0 = 1e-6  # 1 μM
BcrAbl_Imatinib_0 = 0  # No Imatinib-bound BCR-ABL at start

# a few global variables
use_log_scale=1
debug=1
iter=0


# Define ODE system
def bcr_abl_kinetics(t, y):

    global iter
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib = y

    dBcrAbl_active = k_plus1 * BcrAbl_inactive - k_minus1 * BcrAbl_active - k_on1 * ATP_0 * BcrAbl_active + k_off1*BcrAbl_ATP
    dBcrAbl_inactive = k_minus1 * BcrAbl_active - k_plus1 * BcrAbl_inactive - k_on3 * BcrAbl_inactive * Imatinib

    dBcrAbl_ATP = k_on1 * BcrAbl_active * ATP_0 - k_off1 * BcrAbl_ATP

    dImatinib = Kintake - k_on3 * Imatinib * BcrAbl_inactive + k_off3 * BcrAbl_Imatinib
    dBcrAbl_Imatinib = k_on3 * BcrAbl_inactive * Imatinib - k_off3 * BcrAbl_Imatinib

    if debug>0:
        if (iter%10==0):
            print("+------------------------------------------------------------------------------------------------------------------------------------------------------------------+")
            print("|   Time     |      dBcrAbl_active         |     BcrAbl_inactiv          |        BcrAbl_ATP           |        Imatinib             |      BcrAbl_Imatinib        |")
            print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------+")
            print("|            |  Value           Delta      |   Value        Delta        |   Value       Delta         |     Value       Delta       |   Value       Delta         |")
        print("| %10.3f | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e |" %
              (t, BcrAbl_active, dBcrAbl_active, BcrAbl_inactive, dBcrAbl_inactive, BcrAbl_ATP, dBcrAbl_ATP, Imatinib, dImatinib, BcrAbl_Imatinib, dBcrAbl_Imatinib), iter)
    iter+=1

    return [dBcrAbl_active, dBcrAbl_inactive, dBcrAbl_ATP, dImatinib, dBcrAbl_Imatinib]


# Time span for simulation (0 to 100s)
t_end=100
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 500)

# Initial conditions
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0]

'''
‘RK45’ (default): Explicit Runge-Kutta method of order 5(4) [1]. The error is controlled assuming accuracy of the fourth-order method, but steps are taken using the fifth-order accurate formula (local extrapolation is done). A quartic interpolation polynomial is used for the dense output [2]. Can be applied in the complex domain.
‘RK23’: Explicit Runge-Kutta method of order 3(2) [3]. The error is controlled assuming accuracy of the second-order method, but steps are taken using the third-order accurate formula (local extrapolation is done). A cubic Hermite polynomial is used for the dense output. Can be applied in the complex domain.
‘DOP853’: Explicit Runge-Kutta method of order 8 [13]. Python implementation of the “DOP853” algorithm originally written in Fortran [14]. A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output. Can be applied in the complex domain.
‘Radau’: Implicit Runge-Kutta method of the Radau IIA family of order 5 [4]. The error is controlled with a third-order accurate embedded formula. A cubic polynomial which satisfies the collocation conditions is used for the dense output.
‘BDF’: Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation [5]. The implementation follows the one described in [6]. A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification. Can be applied in the complex domain.
‘LSODA’: Adams/BDF method with automatic stiffness detection and switching [7], [8]. This is a wrapper of the Fortran solver from ODEPACK.
'''

# Only the last 3 "Radau","BDF","LSODA" seem to find stable solutions here
algorithms=["RK45","RK23","DOP853","Radau","BDF","LSODA"]

# sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method='LSODA')
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method=algorithms[5])

# retrieve data from solver
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib = sol.y

# calculate active/total ratio
actRatio=BcrAbl_active/(BcrAbl_inactive+BcrAbl_active)

# determine minima and maxima for scaling
max_concentration=max(BcrAbl_active.max(), BcrAbl_inactive.max(), BcrAbl_ATP.max(), Imatinib.max(), BcrAbl_Imatinib.max())
min_concentration=min(BcrAbl_active.min(), BcrAbl_inactive.min(), BcrAbl_ATP.min(), Imatinib.min(), BcrAbl_Imatinib.min())

# Use a secondary y-axis, set x-axis gridlines on
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.grid(axis='x')
ax2 = ax1.twinx()

# set limits
ax1.set_xlim(0, t.max())
ax1.set_ylim(0, max_concentration)
ax2.set_ylim(0,1)

# if a log scale is desired we need a small value for min, cannor be zero, with current data 5e-7 works nicely
if use_log_scale==0:
    ax1.set_ylim(min_concentration, max_concentration)
else:
    ax1.set_ylim(5e-7, max_concentration)
    ax1.set_yscale("log")

# x-axis, 1st y-axis and curves
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Concentration (M)")
ax1.plot(t, BcrAbl_active, label="[Bcr-Abl (Active)]", color="red")
ax1.plot(t, BcrAbl_inactive, label="[Bcr-Abl (Inactive)]", color="blue")
ax1.plot(t, BcrAbl_ATP, label="dBcrAbl_ATP")
ax1.plot(t, Imatinib, label="Imatinib")
ax1.plot(t, BcrAbl_Imatinib, label="BcrAbl_Imatinib")
ax1.legend(loc='lower left')

# 2nd y-axis and curves
ax2.set_ylabel("Proportion Active Bcr-Abl", color='green')
ax2.plot(t, actRatio, label="Active/Inactive Ratio", color="green", linestyle='dashed', lw=1)
ax2.legend(loc='upper right')

plt.title("Kinetic Simulation of BCR-ABL Dynamics (K+1=%5.2f K-1=%5.2f)" % (k_plus1, k_minus1))
plt.grid()
plt.show()
