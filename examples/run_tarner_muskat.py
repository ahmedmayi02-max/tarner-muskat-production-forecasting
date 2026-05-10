import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

N = 145_000_000          # STB
Swi = 0.20               # fraction
m = 0.0                  # no initial gas cap
NUMBER_OF_PRESSURE_DROPS = 26

RF_MIN = 0.05            # 5%
RF_MAX = 0.25            # 25%

END_DATE = "1997-06-01"  # solve only until Jun-97

SAVE_RESULTS_EXCEL = True
SAVE_PLOTS = True
SHOW_INTERSECTION_PLOTS = True
PLOTS_FOLDER = "turner_intersection_plots"


# ============================================================
# 2. PVT DATA EMBEDDED DIRECTLY
# ============================================================

PVT_DATA = [
    # Date, time_months, pressure, Bo, Bg, Rs, mu_o, mu_g
    ("1984-05-01", 12, 5044.0000, 1.970700347, 0.000720000, 1603.000000, 0.4880, 0.0170),
    ("1985-05-01", 24, 4900.0000, 1.979391967, 0.000740000, 1603.000000, 0.5390, 0.0166),
    ("1986-05-01", 36, 4759.0000, 1.961249000, 0.000760093, 1543.500887, 0.5950, 0.0162),
    ("1987-05-01", 48, 4661.0000, 1.940571000, 0.000766773, 1482.169415, 0.6580, 0.0158),
    ("1988-05-01", 60, 4600.0000, 1.927700000, 0.000771401, 1445.151598, 0.7260, 0.0154),
    ("1989-05-01", 72, 4504.0000, 1.907444000, 0.000779447, 1388.635271, 0.8020, 0.0150),
    ("1990-05-01", 84, 4434.0000, 1.892674000, 0.000785923, 1348.729176, 0.8870, 0.0146),
    ("1991-05-01", 96, 4363.0000, 1.877693000, 0.000793039, 1309.342345, 0.9810, 0.0142),
    ("1992-05-01", 108, 4286.0000, 1.861446000, 0.000801402, 1267.832681, 1.0850, 0.0138),
    ("1993-05-01", 120, 4203.0000, 1.843933000, 0.000811202, 1224.452107, 1.1990, 0.0134),
    ("1994-05-01", 132, 4119.0000, 1.826209000, 0.000821988, 1181.944493, 1.3240, 0.0130),
    ("1995-06-01", 144, 4044.0000, 1.810384000, 0.000832390, 1145.140986, 1.4640, 0.0126),
    ("1996-06-01", 156, 3989.0000, 1.798779000, 0.000840502, 1118.822179, 1.6170, 0.0122),
    ("1997-06-01", 168, 3947.0000, 1.789917000, 0.000846981, 1099.097725, 1.6170, 0.0122),
]

pvt = pd.DataFrame(
    PVT_DATA,
    columns=["Date", "time_months", "pressure_psia", "Bo", "Bg", "Rs", "mu_o", "mu_g"]
)

pvt["Date"] = pd.to_datetime(pvt["Date"])
pvt = pvt[pvt["Date"] <= pd.to_datetime(END_DATE)].reset_index(drop=True)


# ============================================================
# 3. KRG/KRO CURVE
# ============================================================

SL_TABLE_PERCENT = np.array([
    30, 40, 50, 60, 70, 80, 85, 88, 90, 91, 92
], dtype=float)

KRG_KRO_TABLE = np.array([
    100, 60, 25, 10, 3, 1, 0.45, 0.20, 0.08, 0.02, 0.001
], dtype=float)


def get_krg_kro(SL):
    """
    SL is total liquid saturation as fraction.
    Example: 0.95 means 95%.
    """

    SL_percent = SL * 100.0

    if SL_percent >= 92:
        return 0.0

    if SL_percent <= SL_TABLE_PERCENT[0]:
        return KRG_KRO_TABLE[0]

    log_kr = np.interp(
        SL_percent,
        SL_TABLE_PERCENT,
        np.log10(KRG_KRO_TABLE)
    )

    return 10 ** log_kr


# ============================================================
# 4. REFERENCE CONDITIONS
# ============================================================

Boi = pvt.loc[0, "Bo"]
Bgi = pvt.loc[0, "Bg"]
Rsi = pvt.loc[0, "Rs"]


# ============================================================
# 5. PHI FUNCTIONS
# ============================================================

def calculate_phi(row):
    Bo = row["Bo"]
    Bg = row["Bg"]
    Rs = row["Rs"]

    denominator = (
        ((Bo - Boi) / Bg)
        + (Rsi - Rs)
        + m * Boi * ((1 / Bgi) - (1 / Bg))
    )

    phi_g = 1.0 / denominator
    phi_o = ((Bo / Bg) - Rs) * phi_g

    return phi_o, phi_g


# ============================================================
# 6. MBE METHOD
# ============================================================

def calculate_mbe_gp(row, Np_prev, Gp_prev, Np_trial):
    delta_Np = Np_trial - Np_prev
    phi_o, phi_g = calculate_phi(row)

    R_dash = (
        N
        - Np_prev * phi_o
        - Gp_prev * phi_g
        - delta_Np * phi_o
    ) / (delta_Np * phi_g)

    delta_Gp = R_dash * delta_Np
    Gp_trial = Gp_prev + delta_Gp

    return {
        "R_dash_MBE": R_dash,
        "delta_Gp_MBE": delta_Gp,
        "Gp_MBE": Gp_trial,
        "phi_o": phi_o,
        "phi_g": phi_g
    }


# ============================================================
# 7. KR METHOD
# ============================================================

def calculate_kr_gp(row, Np_prev, Gp_prev, Np_trial):
    delta_Np = Np_trial - Np_prev

    Bo = row["Bo"]
    Bg = row["Bg"]
    Rs = row["Rs"]
    mu_o = row["mu_o"]
    mu_g = row["mu_g"]

    So = (1 - Swi) * ((N - Np_trial) * Bo) / (N * Boi)
    SL = So + Swi
    krg_kro = get_krg_kro(SL)

    R_dash = Rs + krg_kro * (mu_o / mu_g) * (Bo / Bg)

    delta_Gp = R_dash * delta_Np
    Gp_trial = Gp_prev + delta_Gp

    return {
        "So": So,
        "SL": SL,
        "krg_kro": krg_kro,
        "R_dash_Kr": R_dash,
        "delta_Gp_Kr": delta_Gp,
        "Gp_Kr": Gp_trial
    }


# ============================================================
# 8. INTERSECTION
# ============================================================

def find_intersection(Np_min, Np_max, Gp_MBE_min, Gp_MBE_max, Gp_Kr_min, Gp_Kr_max):
    denominator = (Gp_MBE_max - Gp_MBE_min) - (Gp_Kr_max - Gp_Kr_min)

    if abs(denominator) < 1e-12:
        return np.nan, np.nan, np.nan

    t = (Gp_Kr_min - Gp_MBE_min) / denominator
    Np_actual = Np_min + t * (Np_max - Np_min)
    Gp_actual = Gp_MBE_min + t * (Gp_MBE_max - Gp_MBE_min)

    return Np_actual, Gp_actual, t


# ============================================================
# 8B. MUSKAT METHOD - FIRST YEAR ONLY
# ============================================================

def solve_muskat_first_year():
    """
    Muskat method only for the first pressure drop:
    from 5044 psi to 4900 psi.

    This does not affect the Tarner/Turner prediction.
    It is used only to compare first-year Np.
    """

    row_initial = pvt.loc[0]   # 5044 psi
    row_final = pvt.loc[1]     # 4900 psi

    P_initial = row_initial["pressure_psia"]
    P_final = row_final["pressure_psia"]
    delta_P = P_initial - P_final

    # Initial saturations
    So_initial = 1.0 - Swi
    Sg_initial = 0.0

    # Muskat derivative values from the manual method
    dRs_dP = 0.26
    d_invBg_dP = 0.3853

    # First derivative at 5044 psi
    dSo_dP_initial = (
        So_initial
        * (row_initial["Bg"] / row_initial["Bo"])
        * dRs_dP
        + Sg_initial
        * row_initial["Bg"]
        * d_invBg_dP
    )

    # First estimate of oil saturation at 4900 psi
    So_estimated = So_initial - delta_P * dSo_dP_initial

    # Gas saturation from estimated oil saturation
    Sg_estimated = 1.0 - Swi - So_estimated
    SL_estimated = So_estimated + Swi
    krg_kro_estimated = get_krg_kro(SL_estimated)

    # Second derivative at 4900 psi
    dSo_dP_final = (
        So_estimated
        * (row_final["Bg"] / row_final["Bo"])
        * dRs_dP
        + Sg_estimated
        * row_final["Bg"]
        * d_invBg_dP
    )

    # Average derivative
    dSo_dP_average = 0.5 * (dSo_dP_initial + dSo_dP_final)

    # Final oil saturation at 4900 psi
    So_final = So_estimated - delta_P * dSo_dP_average

    # Calculate Np from saturation equation:
    # So = (1 - Swi) * ((N - Np) * Bo) / (N * Boi)
    Np_Muskat = N * (
        1.0
        - (So_final * Boi)
        / ((1.0 - Swi) * row_final["Bo"])
    )

    return {
        "P_initial": P_initial,
        "P_final": P_final,
        "delta_P": delta_P,

        "So_initial": So_initial,
        "Sg_initial": Sg_initial,

        "dRs_dP": dRs_dP,
        "d_invBg_dP": d_invBg_dP,

        "dSo_dP_initial": dSo_dP_initial,
        "So_estimated": So_estimated,
        "Sg_estimated": Sg_estimated,
        "SL_estimated": SL_estimated,
        "krg_kro_estimated": krg_kro_estimated,

        "dSo_dP_final": dSo_dP_final,
        "dSo_dP_average": dSo_dP_average,
        "So_final": So_final,

        "Np_Muskat_STB": Np_Muskat,
        "Np_Muskat_MMSTB": Np_Muskat / 1e6,
    }


# ============================================================
# 9. MAIN LOOP
# ============================================================

def run_turner_method():
    results = []

    Np_prev = 0.0
    Gp_prev = 0.0

    delta_Np_min = N * RF_MIN / NUMBER_OF_PRESSURE_DROPS
    delta_Np_max = N * RF_MAX / NUMBER_OF_PRESSURE_DROPS

    for i in range(1, len(pvt)):
        row = pvt.loc[i]

        Np_min = Np_prev + delta_Np_min
        Np_max = Np_prev + delta_Np_max

        mbe_min = calculate_mbe_gp(row, Np_prev, Gp_prev, Np_min)
        mbe_max = calculate_mbe_gp(row, Np_prev, Gp_prev, Np_max)

        kr_min = calculate_kr_gp(row, Np_prev, Gp_prev, Np_min)
        kr_max = calculate_kr_gp(row, Np_prev, Gp_prev, Np_max)

        Np_actual, Gp_actual, t = find_intersection(
            Np_min,
            Np_max,
            mbe_min["Gp_MBE"],
            mbe_max["Gp_MBE"],
            kr_min["Gp_Kr"],
            kr_max["Gp_Kr"]
        )

        results.append({
            "step": i,
            "Date": row["Date"],
            "pressure_psia": row["pressure_psia"],

            "Np_prev_STB": Np_prev,
            "Gp_prev_scf": Gp_prev,

            "Np_min_STB": Np_min,
            "Np_max_STB": Np_max,
            "Np_min_MMSTB": Np_min / 1e6,
            "Np_max_MMSTB": Np_max / 1e6,

            "Gp_MBE_min_scf": mbe_min["Gp_MBE"],
            "Gp_MBE_max_scf": mbe_max["Gp_MBE"],
            "Gp_MBE_min_Bscf": mbe_min["Gp_MBE"] / 1e9,
            "Gp_MBE_max_Bscf": mbe_max["Gp_MBE"] / 1e9,

            "Gp_Kr_min_scf": kr_min["Gp_Kr"],
            "Gp_Kr_max_scf": kr_max["Gp_Kr"],
            "Gp_Kr_min_Bscf": kr_min["Gp_Kr"] / 1e9,
            "Gp_Kr_max_Bscf": kr_max["Gp_Kr"] / 1e9,

            "Np_actual_STB": Np_actual,
            "Gp_actual_scf": Gp_actual,
            "Np_actual_MMSTB": Np_actual / 1e6,
            "Gp_actual_Bscf": Gp_actual / 1e9,

            "intersection_t": t,

            # valid_intersection means mathematical intersection exists.
            # inside_original_range means intersection is between Np_min and Np_max.
            "valid_intersection": np.isfinite(t),
            "inside_original_range": np.isfinite(t) and 0 <= t <= 1,

            "phi_o": mbe_min["phi_o"],
            "phi_g": mbe_min["phi_g"],

            "R_dash_MBE_min": mbe_min["R_dash_MBE"],
            "R_dash_MBE_max": mbe_max["R_dash_MBE"],

            "So_min": kr_min["So"],
            "So_max": kr_max["So"],
            "SL_min": kr_min["SL"],
            "SL_max": kr_max["SL"],

            "krg_kro_min": kr_min["krg_kro"],
            "krg_kro_max": kr_max["krg_kro"],

            "R_dash_Kr_min": kr_min["R_dash_Kr"],
            "R_dash_Kr_max": kr_max["R_dash_Kr"],
        })

        Np_prev = Np_actual
        Gp_prev = Gp_actual

    return pd.DataFrame(results)


# ============================================================
# 10. SINGLE INTERSECTION PLOT - EXTENDED LINES
# ============================================================

def plot_intersection(results, step_number, save_plot=False):
    r = results.loc[results["step"] == step_number].iloc[0]

    # Original two trial points
    Np_min = r["Np_min_MMSTB"]
    Np_max = r["Np_max_MMSTB"]

    Gp_MBE_min = r["Gp_MBE_min_Bscf"]
    Gp_MBE_max = r["Gp_MBE_max_Bscf"]

    Gp_Kr_min = r["Gp_Kr_min_Bscf"]
    Gp_Kr_max = r["Gp_Kr_max_Bscf"]

    x_actual = r["Np_actual_MMSTB"]
    y_actual = r["Gp_actual_Bscf"]
    t = r["intersection_t"]

    # Extend the line using the same linear parameter t.
    # t = 0 at Np_min
    # t = 1 at Np_max
    # t > 1 means intersection is to the right
    # t < 0 means intersection is to the left
    if np.isfinite(t):
        t_low = min(0.0, t)
        t_high = max(1.0, t)

        # Add padding so the intersection is not exactly at the figure edge
        padding = 0.10 * (t_high - t_low)
        t_low -= padding
        t_high += padding

        t_plot = np.array([t_low, t_high])
    else:
        t_plot = np.array([0.0, 1.0])

    x_plot = Np_min + t_plot * (Np_max - Np_min)

    y_mbe_plot = Gp_MBE_min + t_plot * (Gp_MBE_max - Gp_MBE_min)
    y_kr_plot = Gp_Kr_min + t_plot * (Gp_Kr_max - Gp_Kr_min)

    # Original short trial points
    x_original = np.array([Np_min, Np_max])
    y_mbe_original = np.array([Gp_MBE_min, Gp_MBE_max])
    y_kr_original = np.array([Gp_Kr_min, Gp_Kr_max])

    plt.figure(figsize=(8, 6))

    # Extended lines
    plt.plot(x_plot, y_mbe_plot, label="MBE extended")
    plt.plot(x_plot, y_kr_plot, label="Kr extended")

    # Original trial points
    plt.scatter(x_original, y_mbe_original, marker="o", label="MBE trial points")
    plt.scatter(x_original, y_kr_original, marker="o", label="Kr trial points")

    # Intersection
    plt.scatter(x_actual, y_actual, s=100, label="Intersection")
    plt.axvline(x_actual, linestyle="--")
    plt.axhline(y_actual, linestyle="--")

    date_str = pd.to_datetime(r["Date"]).strftime("%b-%Y")
    plt.xlabel("Np, MMSTB")
    plt.ylabel("Gp, Bscf")
    plt.title(
        f"Tarner/Turner Intersection - Step {step_number} - {date_str} - "
        f"P = {r['pressure_psia']:.0f} psia"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        filename = os.path.join(PLOTS_FOLDER, f"step_{step_number:02d}_{date_str}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if SHOW_INTERSECTION_PLOTS:
        plt.show()
    else:
        plt.close()


# ============================================================
# 11. ALL INTERSECTION PLOTS
# ============================================================

def plot_all_intersections(results):
    for step_number in results["step"]:
        plot_intersection(results, int(step_number), save_plot=SAVE_PLOTS)


# ============================================================
# 12. FINAL RESULT PLOTS
# ============================================================

def plot_final_results(results):
    plt.figure(figsize=(8, 6))
    plt.plot(results["pressure_psia"], results["Np_actual_MMSTB"], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Pressure, psia")
    plt.ylabel("Np, MMSTB")
    plt.title("Oil Produced vs Pressure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(results["pressure_psia"], results["Gp_actual_Bscf"], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Pressure, psia")
    plt.ylabel("Gp, Bscf")
    plt.title("Gas Produced vs Pressure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 13. RUN
# ============================================================

results = run_turner_method()

# First-year Muskat comparison only
muskat_first = solve_muskat_first_year()

print("\nFIRST PRESSURE DROP VALIDATION")
print("===================================================")

first = results.loc[0]

Np_Tarner = first["Np_actual_STB"]
Np_Muskat = muskat_first["Np_Muskat_STB"]

error_fraction = abs(Np_Muskat - Np_Tarner) / Np_Muskat
error_percent = error_fraction * 100.0

print(f"Date = {pd.to_datetime(first['Date']).strftime('%b-%Y')}")
print(f"Pressure drop = {muskat_first['P_initial']:.0f} psi to {muskat_first['P_final']:.0f} psi")
print(f"Delta P = {muskat_first['delta_P']:.3f} psi")

print("\nTARNER / TURNER METHOD")
print("---------------------------------------------------")
print(f"Np Tarner = {Np_Tarner:.3f} STB")
print(f"Np Tarner = {Np_Tarner / 1e6:.6f} MMSTB")
print(f"Gp Tarner = {first['Gp_actual_scf']:.3f} scf")
print(f"Gp Tarner = {first['Gp_actual_Bscf']:.6f} Bscf")

print("\nMUSKAT METHOD - FIRST YEAR ONLY")
print("---------------------------------------------------")
print(f"So initial = {muskat_first['So_initial']:.6f}")
print(f"Sg initial = {muskat_first['Sg_initial']:.6f}")
print(f"dRs/dP = {muskat_first['dRs_dP']:.6f}")
print(f"d(1/Bg)/dP = {muskat_first['d_invBg_dP']:.6f}")
print(f"dSo/dP initial = {muskat_first['dSo_dP_initial']:.10f}")
print(f"So estimated = {muskat_first['So_estimated']:.6f}")
print(f"Sg estimated = {muskat_first['Sg_estimated']:.6f}")
print(f"SL estimated = {muskat_first['SL_estimated']:.6f}")
print(f"Krg/Kro estimated = {muskat_first['krg_kro_estimated']:.6f}")
print(f"dSo/dP final = {muskat_first['dSo_dP_final']:.10f}")
print(f"dSo/dP average = {muskat_first['dSo_dP_average']:.10f}")
print(f"So final = {muskat_first['So_final']:.6f}")
print(f"Np Muskat = {Np_Muskat:.3f} STB")
print(f"Np Muskat = {Np_Muskat / 1e6:.6f} MMSTB")

print("\nERROR BETWEEN BOTH METHODS - FIRST YEAR ONLY")
print("---------------------------------------------------")
print("Error = |Np Muskat - Np Tarner| / Np Muskat")
print(f"Error = {error_fraction:.6f} fraction")
print(f"Error = {error_percent:.2f} %")

print(f"\nValid mathematical intersection = {first['valid_intersection']}")
print(f"Inside original Np_min/Np_max range = {first['inside_original_range']}")

print("\nFULL RESULTS UNTIL JUN-97")
print("===================================================")
print(results[[
    "step",
    "Date",
    "pressure_psia",
    "Np_actual_MMSTB",
    "Gp_actual_Bscf",
    "valid_intersection",
    "inside_original_range"
]].to_string(index=False))

# Plot every intersection step
plot_all_intersections(results)

# Final plots
plot_final_results(results)

# Save results
if SAVE_RESULTS_EXCEL:
    results.to_excel("turner_results_until_Jun97.xlsx", index=False)
    print("\nSaved results to: turner_results_until_Jun97.xlsx")

print("\nDONE")
