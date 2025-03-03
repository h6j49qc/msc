import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import maximum
from scipy.integrate import solve_ivp

# Define reaction rate constants (assumed reasonable values)
k_plus1 = 0.10  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.0851  # Inactivation rate of BCR-ABL (s⁻¹)
k_prod1 = 5.0e-7
k_prod1 = 0

# Initial concentrations (M)
BcrAbl_active_0 = 2e-6  # 5 μM
BcrAbl_inactive_0 = 5e-6  # 2 μM

# Define ODE system
def bcr_abl_kinetics(t, y):
    BcrAbl_active, BcrAbl_inactive = y

    dBcrAbl_active = k_plus1 * BcrAbl_inactive - k_minus1 * BcrAbl_active
    dBcrAbl_inactive = k_minus1 * BcrAbl_active - k_plus1 * BcrAbl_inactive + k_prod1

    return [dBcrAbl_active, dBcrAbl_inactive]

t_end=40
# Time span for simulation (0 to 1000s)
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 500)

# Initial conditions
y0 = [BcrAbl_active_0, BcrAbl_inactive_0]

sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method='LSODA')
t = sol.t
BcrAbl_active, BcrAbl_inactive = sol.y

print(t)
print(BcrAbl_active)
print(BcrAbl_inactive)
actRatio=BcrAbl_active/(BcrAbl_active+BcrAbl_inactive)
print(actRatio)

# Use a secondary y-axis, set x-axis gridlines on
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.grid(axis='x')
ax2 = ax1.twinx()

# set limits
ax1.set_xlim(0, t.max())
ax1.set_ylim(0, max(BcrAbl_active.max(), BcrAbl_inactive.max()))
ax2.set_ylim(0,1)

# 1st y-axis and curves
ax1.set_ylabel("Concentration (M)")
ax1.plot(t, BcrAbl_active, label="[Bcr-Abl (Active)]", color="red")
ax1.plot(t, BcrAbl_inactive, label="[Bcr-Abl (Inactive)]", color="blue")
ax1.legend(loc='lower left')
ax1.set_xlabel("Time (s)")

# 2nd y-axis and curves
ax2.set_ylabel("Proportion Active Bcr-Abl", color='green')
ax2.plot(t, actRatio, label="Active/Inactive Ratio", color="green", linestyle='dashed', lw=1)
ax2.legend(loc='upper right')

plt.title("Kinetic Simulation of BCR-ABL Dynamics K+1=%5.2f K-1=%5.2f" % (k_plus1, k_minus1))
plt.grid()
plt.show()