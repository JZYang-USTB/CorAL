import numpy as np
from scipy.optimize import least_squares


DEMONSTRATION_CIRCUITS = {
    "one_r_cpe": (1, False),
    "two_r_cpe": (2, False),
    "one_r_cpe_with_warburg": (1, True),
}
ALTERNATIVE_CIRCUITS = {"two_r_cpe_with_warburg": (2, True)}
CHI_SQUARE_THRESHOLD = 1000.0


def _spectrum(parameters, frequency, arcs, diffusion):
    z = np.full(len(frequency), parameters[0], dtype=complex)
    jw = 2j * np.pi * frequency
    for index in range(arcs):
        resistance, q, exponent = parameters[1 + 3 * index:4 + 3 * index]
        branch = resistance
        if diffusion and index == arcs - 1:
            branch = branch + parameters[-1] / np.sqrt(jw)
        z += 1.0 / (1.0 / branch + q * jw ** exponent)
    return z


def _fit(frequency, measured, arcs, diffusion):
    magnitude = np.maximum(np.abs(measured), 1e-12)
    guess = [max(float(measured.real.min()), 1e-4)]
    lower, upper = [1e-8], [1e9]
    for index in range(arcs):
        guess.extend([max(float(np.ptp(measured.real)) / arcs, 1.0),
                      10.0 ** (-4 - index), 0.85])
        lower.extend([1e-8, 1e-14, 0.05])
        upper.extend([1e9, 10, 1])
    if diffusion:
        guess.append(10.0)
        lower.append(1e-8)
        upper.append(1e9)
    initial, minimum, maximum = (np.log(values) for values in (guess, lower, upper))

    def residual(log_parameters):
        error = (_spectrum(np.exp(log_parameters), frequency, arcs, diffusion) - measured) / magnitude
        return np.column_stack((error.real, error.imag)).ravel()

    fit = least_squares(residual, np.clip(initial, minimum, maximum), bounds=(minimum, maximum), max_nfev=2500)
    if not fit.success:
        return {"status": "fit_failed"}
    parameters = np.exp(fit.x)
    names = ["Rs"] + [f"{key}{index}" for index in range(1, arcs + 1) for key in ("R", "Q", "n")]
    if diffusion:
        names.append("warburg_coefficient")
    prediction = _spectrum(parameters, frequency, arcs, diffusion)
    return {"status": "fitted", "chi_square": float(fit.fun @ fit.fun),
            "parameters": dict(zip(names, parameters.tolist(), strict=True)),
            "fitted_real_ohm": prediction.real.tolist(), "fitted_imag_ohm": prediction.imag.tolist()}


def fit_eis(frequency, real, imaginary, *, threshold=CHI_SQUARE_THRESHOLD, plot_path=None):
    frequency = np.asarray(frequency, dtype=float)
    measured = np.asarray(real, dtype=float) + 1j * np.asarray(imaginary, dtype=float)
    if (frequency.ndim != 1 or frequency.shape != measured.shape or len(frequency) < 8
            or not np.isfinite(frequency).all() or not np.isfinite(measured).all()
            or (frequency <= 0).any() or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("At least eight finite EIS observations and a positive threshold are required.")
    attempts = {}
    selected = None
    for candidates in (DEMONSTRATION_CIRCUITS, ALTERNATIVE_CIRCUITS):
        for name, (arcs, diffusion) in candidates.items():
            try:
                attempts[name] = _fit(frequency, measured, arcs, diffusion)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                attempts[name] = {"status": "fit_failed"}
        accepted = [name for name in candidates if attempts[name].get("chi_square", np.inf) < threshold]
        if accepted:
            selected = min(accepted, key=lambda name: attempts[name]["chi_square"])
            break
    if selected and plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        best = attempts[selected]
        order = np.argsort(frequency)
        fig, ax = plt.subplots()
        ax.scatter(measured.real, -measured.imag, label="Spectrum")
        ax.plot(np.array(best["fitted_real_ohm"])[order], -np.array(best["fitted_imag_ohm"])[order], label=selected)
        ax.set(xlabel="Re(Z) (ohm)", ylabel="-Im(Z) (ohm)")
        ax.legend()
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return {"status": "accepted" if selected else "remeasurement_required",
            "threshold": threshold, "selected_circuit": selected, "attempts": attempts}


def inhibition_efficiency_at_001_hz(inhibited, blank):
    def modulus(spectrum):
        f, real, imaginary = np.asarray(spectrum, dtype=float).T
        matches = np.flatnonzero(np.isclose(f, 0.01, rtol=1e-5, atol=0))
        if len(matches) != 1 or not np.isfinite([real[matches[0]], imaginary[matches[0]]]).all():
            raise ValueError("A unique finite impedance observation at 0.01 Hz is required.")
        index = matches[0]
        return float(np.hypot(real[index], imaginary[index]))

    measured, control = modulus(inhibited), modulus(blank)
    if measured <= 0 or control <= 0:
        raise ValueError("Impedance moduli must be positive.")
    return 100 * (measured - control) / measured


def synergy_index(measured_ie, fractions, component_ie):
    weights, observed = np.asarray(fractions), np.asarray(component_ie)
    if (weights.ndim != 1 or weights.shape != observed.shape or not np.isfinite(weights).all()
            or not np.isfinite(observed).all() or not np.isfinite(measured_ie)
            or not np.isclose(weights.sum(), 1) or (weights < 0).any()):
        raise ValueError("Supply aligned finite efficiencies and normalized nonnegative fractions.")
    weighted = float(weights @ observed)
    return {"weighted_ie_pct": weighted,
            "synergy_index_pct": 100 * (measured_ie - weighted) / weighted if weighted > 0 else None}
