#!/usr/bin/env python3
"""
Interactive discrete map explorer (terminal prompts, no CLI flags).

Maps included
-------------
1) Logistic:          x_{t+1} = r x_t (1 - x_t),       r in [0, 4]
2) Tent-like (piecewise):
                      f(x) = μ x          for 0 ≤ x ≤ 1/2
                           = μ (1 - x)    for 1/2 < x ≤ 1,  μ in [0, 2]

Features
--------
• Time series
• Cobweb diagram
• Bifurcation diagram

Notes
-----
• State clipped into [0, 1] to avoid numerical drift.
• Sensible defaults are suggested in the prompts; just press Enter to accept.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


# -------------------------- Map definitions --------------------------

def logistic_map(x: np.ndarray | float, r: float) -> np.ndarray | float:
    x = np.asarray(x)
    y = r * x * (1.0 - x)
    return np.clip(y, 0.0, 1.0)

def tent_map(x: np.ndarray | float, mu: float) -> np.ndarray | float:
    x = np.asarray(x)
    y = np.where(x <= 0.5, mu * x, mu * (1.0 - x))
    return np.clip(y, 0.0, 1.0)


@dataclass
class MapSpec:
    name: str
    param_name: str
    param_symbol: str
    param_default: float
    param_min: float
    param_max: float
    func: Callable[[np.ndarray | float, float], np.ndarray | float]
    # Bifurcation sweep defaults
    bif_sweep_min: float
    bif_sweep_max: float
    bif_sweep_pts: int
    bif_transients: int
    bif_keep: int


LOGISTIC = MapSpec(
    name="Logistic",
    param_name="r",
    param_symbol="r",
    param_default=3.7,
    param_min=0.0,
    param_max=4.0,
    func=logistic_map,
    bif_sweep_min=0.0,
    bif_sweep_max=4.0,
    bif_sweep_pts=1200,
    bif_transients=500,
    bif_keep=300,
)

TENT = MapSpec(
    name="Tent-like (piecewise)",
    param_name="mu",
    param_symbol="μ",
    param_default=1.8,
    param_min=0.0,
    param_max=2.0,
    func=tent_map,
    bif_sweep_min=0.0,
    bif_sweep_max=2.0,
    bif_sweep_pts=1200,
    bif_transients=500,
    bif_keep=300,
)


# -------------------------- Core helpers --------------------------

def iterate_map(map_func: Callable[[float, float], float], x0: float, param: float, steps: int) -> np.ndarray:
    xs = np.empty(steps + 1, dtype=float)
    xs[0] = np.clip(x0, 0.0, 1.0)
    for t in range(steps):
        xs[t+1] = map_func(xs[t], param)
    return xs

def bifurcation_pairs(
    map_func: Callable[[float, float], float],
    sweep_min: float, sweep_max: float, sweep_pts: int,
    transients: int, keep: int, x0: float
) -> Tuple[np.ndarray, np.ndarray]:
    params = np.linspace(sweep_min, sweep_max, sweep_pts)
    x = x0
    xs_list: List[np.ndarray] = []
    ps_list: List[np.ndarray] = []
    for p in params:
        x = (x + 1e-6) % 1.0  # de-synchronize
        seq = iterate_map(map_func, x, p, transients + keep)
        tail = seq[-keep:]
        xs_list.append(tail)
        ps_list.append(np.full_like(tail, p))
        x = tail[-1]
    return np.concatenate(ps_list), np.concatenate(xs_list)


# -------------------------- Plotting --------------------------

def plot_time_series(xs: np.ndarray, param_value: float, param_symbol: str, map_name: str) -> None:
    plt.figure()
    plt.plot(np.arange(xs.size), xs, marker=".", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("x_t")
    plt.title(f"{map_name} time series ({param_symbol}={param_value:g})")
    plt.tight_layout()

def plot_cobweb(map_func, param_value: float, x0: float, param_symbol: str, map_name: str,
                steps: int = 100, dense: int = 1000) -> None:
    grid = np.linspace(0, 1, dense)
    fx = map_func(grid, param_value)

    plt.figure()
    plt.plot(grid, fx, label="map")
    plt.plot(grid, grid, linestyle="--", label="y = x")

    x = x0
    for _ in range(steps):
        y = map_func(x, param_value)
        plt.plot([x, x], [x, y], linewidth=0.8)  # vertical
        plt.plot([x, y], [y, y], linewidth=0.8)  # horizontal
        x = y

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{map_name} cobweb ({param_symbol}={param_value:g}, x0={x0:g})")
    plt.legend()
    plt.tight_layout()

def plot_bifurcation_generic(
    map_func,
    sweep_min: float, sweep_max: float, sweep_pts: int,
    transients: int, keep: int, x0: float,
    param_name: str, param_symbol: str, map_name: str,
    vlines: List[float] | None = None,
    font_size: int = 14
) -> None:
    ps, xs = bifurcation_pairs(map_func, sweep_min, sweep_max, sweep_pts, transients, keep, x0)

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(ps, xs, marker=".", linestyle="", markersize=0.6)
    plt.xlabel(param_symbol, fontsize=font_size)
    plt.ylabel("x", fontsize=font_size)
    plt.title(f"{map_name} bifurcation diagram", fontsize=font_size + 2)
    plt.tick_params(axis="both", which="major", labelsize=font_size - 2)

    if vlines:
        for pv in vlines:
            if sweep_min <= pv <= sweep_max:
                plt.axvline(x=pv, linestyle="--", linewidth=1)

    plt.tight_layout()


# -------------------------- Input utilities --------------------------

def ask_choice(prompt: str, choices: dict[str, object], default_key: str) -> object:
    keys = list(choices.keys())
    show = " / ".join([f"{k}" + ("*" if k == default_key else "") for k in keys])
    while True:
        s = input(f"{prompt} [{show}]: ").strip().lower()
        if s == "" and default_key in choices:
            return choices[default_key]
        if s in choices:
            return choices[s]
        print(f"Please choose one of: {', '.join(keys)}")

def ask_float(prompt: str, default: float, min_val: float | None = None, max_val: float | None = None) -> float:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            val = default
        else:
            try:
                val = float(s)
            except ValueError:
                print("Enter a number.")
                continue
        if min_val is not None and val < min_val:
            print(f"Must be ≥ {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Must be ≤ {max_val}.")
            continue
        return val

def ask_int(prompt: str, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            val = default
        else:
            try:
                val = int(s)
            except ValueError:
                print("Enter an integer.")
                continue
        if min_val is not None and val < min_val:
            print(f"Must be ≥ {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Must be ≤ {max_val}.")
            continue
        return val

def ask_bool(prompt: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} ({d}) ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes"): return True
        if s in ("n", "no"):  return False
        print("Please answer y or n.")


# -------------------------- Interactive flow --------------------------

def main():
    print("Choose a map:")
    print("\n=== Discrete Map Explorer ===")
    choices = {
        "l": LOGISTIC,  # logistic
        "t": TENT,      # tent
    }
    spec: MapSpec = ask_choice("Map (l=logistic, t=tent)", choices, default_key="l")

    # Core parameters
    x0   = ask_float("Initial condition x0 in [0,1]", 0.12345, 0.0, 1.0)
    param = ask_float(
        f"{spec.param_name} ({spec.param_symbol})",
        default=spec.param_default,
        min_val=spec.param_min,
        max_val=spec.param_max
    )
    steps = ask_int("Number of iteration steps", 200, 1, 10_000)

    # What to plot?
    print("Defaults are Yes.")
    do_ts     = ask_bool("Plot time series?", True)
    do_cob    = ask_bool("Plot cobweb diagram?", True)
    do_bif    = ask_bool("Plot bifurcation diagram?", True)

    # Time series + cobweb
    xs = iterate_map(spec.func, x0, param, steps)
    print(f"\nParameters: {spec.param_name}={param:g}, x0={x0:g}, steps={steps}")
    print("First 10 states:", np.array2string(xs[:10], precision=6, separator=", "))

    if do_ts:
        plot_time_series(xs, param, spec.param_symbol, spec.name)
    if do_cob:
        plot_cobweb(spec.func, param, x0, spec.param_symbol, spec.name, steps=min(steps, 200))

    # Bifurcation options
    if do_bif:
        sweep_min = ask_float(f"Bifurcation: {spec.param_name}_min", spec.bif_sweep_min, spec.param_min, spec.param_max)
        sweep_max = ask_float(f"Bifurcation: {spec.param_name}_max", spec.bif_sweep_max, spec.param_min, spec.param_max)
        if sweep_max < sweep_min:
            sweep_min, sweep_max = sweep_max, sweep_min
        sweep_pts = ask_int("Bifurcation: number of parameter samples", spec.bif_sweep_pts, 100, 100_000)
        transients = ask_int("Bifurcation: transient iterations to drop", spec.bif_transients, 0, 100_000)
        keep = ask_int("Bifurcation: points per parameter to keep/show", spec.bif_keep, 50, 100_000)

        vlines: List[float] = []
        if spec is LOGISTIC:
            # Offer handy reference lines for logistic map
            if ask_bool("Add vertical guides at common logistic r-values (e.g., r=3, ~3.56995, ~3.828)?", True):
                vlines = [3.0, 3.56995, 3.828]  # pitchfork onset, Feigenbaum point (~), onset of chaos window marker
        else:
            # Optionally ask user for custom lines
            if ask_bool("Add custom vertical guide lines?", False):
                raw = input(f"Enter {spec.param_name}-values separated by spaces: ").strip()
                if raw:
                    for tok in raw.split():
                        try:
                            vlines.append(float(tok))
                        except ValueError:
                            pass

        plot_bifurcation_generic(
            spec.func,
            sweep_min, sweep_max, sweep_pts,
            transients, keep, x0,
            param_name=spec.param_name,
            param_symbol=spec.param_symbol,
            map_name=spec.name,
            vlines=vlines
        )

    if do_ts or do_cob or do_bif:
        plt.show()
    else:
        print("\n(No plots selected.)")

if __name__ == "__main__":
    main()
