# ============================================================
# ---------------main.py-----------------------------------
#
#   query -> download -> analysis (BLS) -> pre_model -> model -> comparison
#
# Each stage's functions are documented in their own file; this file
# only sequences them and passes outputs to the next stage's inputs.
# ============================================================

from query import query_hot_jupiters, get_user_input
from download import select_observation, get_quality_choice, download_selected
from analysis import (
    estimate_period, get_robust_duration, save_transit_solution,
    get_transit_window, check_transit_minimum,
    plot_bls_periodogram, plot_folded_transit,
)
from pre_model import prepare_model_inputs
from model import run_transit_model, plot_transit_model
from comparison import offer_archive_comparison


def main():
    # ---------------- 1. query ----------------
    hot_jupiters = query_hot_jupiters()
    planet_name, mission, stellar_mass, stellar_radius = get_user_input(hot_jupiters)

    # ---------------- 2. download ----------------
    search_result, idx = select_observation(planet_name, mission)
    quality_bitmask = get_quality_choice()
    lc = download_selected(search_result, idx, quality_bitmask=quality_bitmask)

    # ---------------- 3. analysis (BLS period/duration search) ----------------
    best_period, best_t0, best_dur, bls, results_fine = estimate_period(lc)

    final_dur_hr, final_t0, final_depth, snr, source, a_Rs = get_robust_duration(
        lc, bls, best_period, best_t0, best_dur,
        planet_name, stellar_mass, stellar_radius
    )

    solution = save_transit_solution(
        planet_name, best_period, final_t0, final_dur_hr, a_Rs, final_depth, snr, source
    )
    plot_bls_periodogram(results_fine, best_period, planet_name)

    transit_window = get_transit_window(lc, solution)
    check_transit_minimum(transit_window, solution)
    plot_folded_transit(transit_window, solution)

    # ---------------- 4. pre-model ----------------
    pre_model_result = prepare_model_inputs(transit_window, solution, stellar_radius)

    # ---------------- 5. model (Bayesian fit + SNR) ----------------
    model_result = run_transit_model(transit_window, solution, pre_model_result)
    plot_transit_model(model_result, planet_name=planet_name)

    # ---------------- 6. comparison ----------------
    offer_archive_comparison(planet_name, solution, model_result)

    return {
        "solution": solution,
        "pre_model_result": pre_model_result,
        "model_result": model_result,
    }


if __name__ == "__main__":
    main()