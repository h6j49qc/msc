import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.ticker as mticker

# Define reaction rate constants (assumed reasonable values)
k_on1 = 1e2  # ATP binding rate (M⁻¹s⁻¹)
k_off1 = 0.1  # ATP unbinding rate (s⁻¹)
k_on3 = 1e2  # Imatinib binding rate (M⁻¹s⁻¹)
k_off3 = 0.00  # Imatinib unbinding rate (s⁻¹)
k_cat = 0.5  # Catalytic phosphorylation rate (s⁻¹)

k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.0851  # Inactivation rate of BCR-ABL (s⁻¹)
# k_minus1 = 0.014  # Inactivation rate of BCR-ABL (s⁻¹) - rate for mutant variant

Kintake = 3e-10 # Imatinib intake Ms⁻¹ (dosage 600mg or approx. 1.2 millimole daily)

k_on2 = 1.0e6  # Substrate binding rate (M⁻¹s⁻¹)
k_off2 = 0.1  # Substrate unbinding rate (s⁻¹)
k_prod1 = 1e-8
k_deg = 1e-1

# Initial concentrations (M)
BcrAbl_active_0 = 5e-6  # 5 μM
BcrAbl_inactive_0 = 2e-6  # 2 μM
ATP_0 = 1.1e-3  # 1 mM (intracellular ATP concentration)
BcrAbl_ATP_0 = 0  # No bound ATP at start
Imatinib_0 = 1e-6  # 1 μM
BcrAbl_Imatinib_0 = 0  # No Imatinib-bound BCR-ABL at start

Substrate_0 = 1e-6  # 10 μM
BcrAbl_Substrate_0 = 0  # No bound substrate at start
Phospho_Substrate_0 = 0  # Initially no phosphorylated substrate


# Define ODE system
def bcr_abl_kinetics(t, y):

    # need to declare iter as global because we change it here, so python otherwise assumes it is local
    global iter
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = y

    dBcrAbl_active = k_plus1 * BcrAbl_inactive - k_minus1 * BcrAbl_active - k_on1 * ATP_0 * BcrAbl_active + k_off1*BcrAbl_ATP
    dBcrAbl_inactive = k_minus1 * BcrAbl_active - k_plus1 * BcrAbl_inactive - k_on3 * BcrAbl_inactive * Imatinib + k_prod1

    dBcrAbl_ATP = k_on1 * BcrAbl_active * ATP_0 - k_off1 * BcrAbl_ATP + k_off2 * BcrAbl_Substrate

    dImatinib = Kintake - k_on3 * Imatinib * BcrAbl_inactive + k_off3 * BcrAbl_Imatinib
    dBcrAbl_Imatinib = k_on3 * BcrAbl_inactive * Imatinib - k_off3 * BcrAbl_Imatinib - k_deg * BcrAbl_Imatinib

    dSubstrate =  k_off2 * BcrAbl_Substrate - k_on2 * Substrate * dBcrAbl_ATP
    # try treating this as a constant as we have no replenishment process
#    dSubstrate = 0
    dBcrAbl_Substrate = k_on2 * Substrate * dBcrAbl_ATP - k_off2 * BcrAbl_Substrate - k_cat * BcrAbl_Substrate
    dPhospho_Substrate = k_cat * BcrAbl_Substrate

    if debug>0:
        if (iter%10==0):
            print("+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+")
            print("|   Time     |      dBcrAbl_active         |     BcrAbl_inactiv          |        BcrAbl_ATP           |        Imatinib             |      BcrAbl_Imatinib        |        Substrate            |       BcrAbl_Substrate      |      Phospho_Substrate      | Iteration |")
            print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------|")
            print("|            |  Value           Delta      |   Value        Delta        |   Value       Delta         |     Value       Delta       |   Value       Delta         |     Value       Delta       |   Value       Delta         |   Value       Delta         |           |")
        print("| %10.3f | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e |  %6i   |" %
              (t, BcrAbl_active, dBcrAbl_active, BcrAbl_inactive, dBcrAbl_inactive, BcrAbl_ATP, dBcrAbl_ATP, Imatinib, dImatinib,
                  BcrAbl_Imatinib, dBcrAbl_Imatinib, Substrate, dSubstrate, BcrAbl_Substrate, dBcrAbl_Substrate, Phospho_Substrate, dPhospho_Substrate, iter))
    iter+=1
    return [dBcrAbl_active, dBcrAbl_inactive, dBcrAbl_ATP, dImatinib, dBcrAbl_Imatinib, dSubstrate, dBcrAbl_Substrate, dPhospho_Substrate]

def drawPlot(xValues, y1Values, y1Labels, y2Values, y2Labels, title, y1LogScale=False, y1LegendPosition="lower center"):

    # determine minima and maxima for scaling
    minY1, maxY1, minY2, maxY2 = 1e20, 0.0, 1e20, 0.0
    for y1 in y1Values:
        if y1LogScale:
            minY1=min(minY1, min(x for x in y1 if x > 0))
        else:
            minY1=min(minY1, y1.min())
            minY1=0
        maxY1=max(maxY1, y1.max())
    for y2 in y2Values:
        minY2=min(minY2, y2.min())
        maxY2=max(maxY2, y2.max())

    if debug!=0:
        print("y1 min,max = %10.2e, %10.2e"%(minY1, maxY1))
    # Use a secondary y-axis, set x-axis gridlines on
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.grid(axis='x')

    ax1.set_xlim(0, xValues.max())
    ax1.set_ylim(minY1, maxY1*1.05)
    if (y1LogScale):
        ax1.set_yscale("log")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Concentration (M)")
    i=0
    for y1 in y1Values:
        print(y1Labels[i], len(y1Values))
        if (i==0):
            ax1.plot(t, y1, label=y1Labels[i], linestyle='solid', lw=2)
        else:
            ax1.plot(t, y1, label=y1Labels[i], linestyle='solid', lw=2)
        i+=1
    ax1.legend(loc=y1LegendPosition)

    if len(y2Values)>0:
        ax2 = ax1.twinx()
        ax2.set_ylim(minY2, maxY2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Proportion Active Bcr-Abl", color='green')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
        i=0
        for y2 in y2Values:
            ax2.plot(t, y2, label=y2Labels[i], linestyle='dashed', lw=2)
            i+=1
        ax2.legend(loc='upper right')

    plt.title(title)
    plt.grid()
    plt.grid(axis='x')

# ======================================
# Main Graph Generation code starts here
# ======================================
# Only the last 3 "Radau","BDF","LSODA" seem to find stable solutions here,
# unless max_step is reduced from default. See below for documentation
algorithms=["RK45","RK23","DOP853","Radau","BDF","LSODA"]
algo=5

# a few global variables
iter=0
use_log_scale=False
debug=0

# Time span for simulation (0 to 100s)
t_end=80
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 500)


# ======================================================================
# GRAPH 1 #
# 1: 2 Graphs (free wt and mutant Bcr-Abl) Proportion
# Title: Free BCR-ABL Equilibrium WT / Free BCR-ABL Equilibrium Mutant
# ======================================================================

y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
k_minus1=0.0851
iter = 0
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
y2Values.append(BcrAbl_active / (BcrAbl_active + BcrAbl_inactive))
y1Values.append(BcrAbl_active)
y1Values.append(BcrAbl_inactive)
y1Labels=["[BcrAbl_active] (Wild)", "[BcrAbl_inactive] (Wild)"]
y2Labels=["Proportion Active BcrAbl (Wild)"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Free BCR-ABL Equilibrium WT", use_log_scale)

y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
k_minus1= 0.014
iter = 0
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
y2Values.append(BcrAbl_active / (BcrAbl_active + BcrAbl_inactive))
y1Values.append(BcrAbl_active)
y1Values.append(BcrAbl_inactive)
y1Labels=["[BcrAbl_active] (Mutant)", "[BcrAbl_inactive] (Mutant)"]
y2Labels=["Proportion Active BcrAbl (Mutant)"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Free BCR-ABL Equilibrium Mutant", use_log_scale)

# ======================================================================
# GRAPH 2
# 2 Graphs (wt and mutant Bcr-Abl) displaying ALL Components of the Model
# Title: Development of All Model Components WT / Mutant
# This has been duplicated to repeat on log scale so 4 graphs total here
# ======================================================================

loop=0
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
    type="WT"
    if loop>0:
        type="Mutant"
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y2Values.append(BcrAbl_active/(BcrAbl_active+BcrAbl_inactive))
    y1Values.append(BcrAbl_active)
    y1Values.append(BcrAbl_inactive)
    y1Values.append(BcrAbl_ATP)
    y1Values.append(Imatinib)
    y1Values.append(BcrAbl_Imatinib)
    y1Values.append(Substrate)
    y1Values.append(BcrAbl_Substrate)
    y1Values.append(Phospho_Substrate)

    y1Labels=["[BcrAbl_active]", "[BcrAbl_inactive]", "[BcrAbl_ATP]", "[Imatinib]", "[BcrAbl_Imatinib]",
              "[Substrate]", "[BcrAbl_Substrate]","[Phospho_Substrate]"]
    y2Labels=["Proportion Active BcrAbl"]
    drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Time evolution of All Model Components %s"%type, use_log_scale)
    loop+=1

use_log_scale=True
loop=0
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
    type="WT"
    if loop>0:
        type="Mutant"
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y2Values.append(BcrAbl_active/(BcrAbl_active+BcrAbl_inactive))
    y1Values.append(BcrAbl_active)
    y1Values.append(BcrAbl_inactive)
    y1Values.append(BcrAbl_ATP)
    y1Values.append(Imatinib)
    y1Values.append(BcrAbl_Imatinib)
    y1Values.append(Substrate)
    y1Values.append(BcrAbl_Substrate)
    y1Values.append(Phospho_Substrate)

    y1Labels=["[BcrAbl_active]", "[BcrAbl_inactive]", "[BcrAbl_ATP]", "[Imatinib]", "[BcrAbl_Imatinib]",
              "[Substrate]", "[BcrAbl_Substrate]","[Phospho_Substrate]"]
    y2Labels=["Proportion Active BcrAbl"]
    drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Time evolution of All Model Components %s"%type, use_log_scale)
    loop+=1

# ======================================================================
# GRAPH 3 #
# 1 Graph displaying only the wt vs mutant substrate-phospho rate
# (pls call this Active Substrate in the Graph) and the
# bcr-abl-imantinib bound rate of the Model
# Title: Development of Relevant Components
# ======================================================================

use_log_scale=False
loop=0
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    type="WT"
    if loop>0:
        type="Mutant"
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(BcrAbl_Imatinib)
    y1Labels.append("%s Bound to Imatinib"%type)
    loop+=1
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Time Evolution of Bound Imatinib Concentration", use_log_scale)

loop=0
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    type="WT"
    if loop>0:
        type="Mutant"
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(Phospho_Substrate)
    y1Labels.append("%s Active Substrate"%type)
    loop+=1
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Time Evolution of Substrate Concentration", use_log_scale)


# ======================================================================
# GRAPH 4 #
# Apoptosis Rate in one Graph for wt vs mutant
# Title: Apoptosis Rate WT vs Mutant
# This graph is identical to 1st one above in 3 except for constant Kdeg,
# so possibly we will leave this out
# ======================================================================

loop=0
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(BcrAbl_Imatinib*k_deg)

y1Labels=["Wild", "Mutant"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Apoptosis Rate", False)


# ======================================================================
# GRAPH 5 #
# Graph that indicates that 4x imatinib conc. is nessecary in the
# mutant to achieve wt effectivity results
# Title: Adjustment of Imantinib Concentration in Mutant
# ======================================================================

y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.0851  # Inactivation rate of BCR-ABL (s⁻¹)
Imatinib_0 = 1e-6  # 1 μM
iter = 0
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
y1Values.append(BcrAbl_Imatinib)
y1Labels.append("Wild BcrAbl (Imatinib = %iμM)" % (Imatinib_0*1e6))
#y2Values.append(BcrAbl_active / (BcrAbl_active + BcrAbl_inactive))
#y2Labels.append("Wild BcrAbl, proportion [BcrAbl_active]")

k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.014  # Inactivation rate of BCR-ABL (s⁻¹)

for Imatinib_0 in [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]:
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(BcrAbl_Imatinib)
    y1Labels.append("Mutant BcrAbl (Imatinib = %iμM)" % (Imatinib_0*1e6))
#    if len(y2Values)<2:
#        y2Values.append(BcrAbl_active/(BcrAbl_active+BcrAbl_inactive))
#        y2Labels.append("Mutant BcrAbl, proportion [BcrAbl_active]")

drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "[BcrAbl_Imatinib] by Dosage, Time Dependemcy", False, "upper right")

plt.show()



'''
# some old code archived here
use_new_params=False
if use_new_params:
    k_on1 = 5000  # ATP binding rate (M⁻¹s⁻¹)
    k_off1 = 0.25  # ATP unbinding rate (s⁻¹)
    k_on3 = 0.36e6  # Imatinib binding rate (M⁻¹s⁻¹)
    k_off3 = 0.1  # Imatinib unbinding rate (s⁻¹)
    k_cat = 0.71  # Catalytic phosphorylation rate (s⁻¹)


y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
k_minus1=0.014  # Inactivation rate of BCR-ABL (s⁻¹)
iter = 0
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
y1Values.append(BcrAbl_inactive)
y1Values.append(Phospho_Substrate)
y1Values.append(BcrAbl_Imatinib)
y1Labels=["Mutant BcrAbl_inactive", "Mutant Phospho", "Mutant Bound"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Title1", use_log_scale)


k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.0851  # Inactivation rate of BCR-ABL (s⁻¹)
Imatinib_0 = 1e-6  # 1 μM
iter = 0
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
y1Values.append(BcrAbl_Imatinib)
y1Labels.append("Wild BcrAbl (Imatinib = %iμM)" % (Imatinib_0*1e6))
y2Values.append(BcrAbl_active / (BcrAbl_active + BcrAbl_inactive))
y2Labels.append("Wild BcrAbl, proportion [BcrAbl_active]")

k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.014  # Inactivation rate of BCR-ABL (s⁻¹)

for Imatinib_0 in [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]:
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(BcrAbl_Imatinib)
    y1Labels.append("Mutant BcrAbl (Imatinib = %iμM)" % (Imatinib_0*1e6))
    if len(y2Values)<2:
        y2Values.append(BcrAbl_active/(BcrAbl_active+BcrAbl_inactive))
        y2Labels.append("Mutant BcrAbl, proportion [BcrAbl_active]")

drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "[BcrAbl_Imatinib] by Dosage, Time Dependemcy", True)




k_on1 = 5000  # ATP binding rate (M⁻¹s⁻¹)
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(Phospho_Substrate)
    y1Values.append(BcrAbl_Imatinib)
y1Labels=["Wild Phospho", "Wild Bound", "Mutant Phospho", "Mutant Bound"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Title2 k_on1 = 5000", use_log_scale)

k_off1 = 0.25  # ATP unbinding rate (s⁻¹)
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(Phospho_Substrate)
    y1Values.append(BcrAbl_Imatinib)
y1Labels=["Wild Phospho", "Wild Bound", "Mutant Phospho", "Mutant Bound"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Title3 k_off1 = 0.25", use_log_scale)

k_on3 = 0.36e6  # Imatinib binding rate (M⁻¹s⁻¹)
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(Phospho_Substrate)
    y1Values.append(BcrAbl_Imatinib)
y1Labels=["Wild Phospho", "Wild Bound", "Mutant Phospho", "Mutant Bound"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Title4 k_on3 = 0.36e6", use_log_scale)

k_off3 = 0.1  # Imatinib unbinding rate (s⁻¹)
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(Phospho_Substrate)
    y1Values.append(BcrAbl_Imatinib)
y1Labels=["Wild Phospho", "Wild Bound", "Mutant Phospho", "Mutant Bound"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Title5 k_off3 = 0.1", use_log_scale)

k_cat = 0.71  # Catalytic phosphorylation rate (s⁻¹)
y1Values, y1Labels, y2Values, y2Labels = ([], [], [], [])
for k_minus1 in (0.0851, 0.014):  # Inactivation rate of BCR-ABL (s⁻¹)
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    t = sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(Phospho_Substrate)
    y1Values.append(BcrAbl_Imatinib)
y1Labels=["Wild Phospho", "Wild Bound", "Mutant Phospho", "Mutant Bound"]
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "Title6 k_cat = 0.71", use_log_scale)


'''