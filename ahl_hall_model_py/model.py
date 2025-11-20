import numpy as np

from ahl_hall_model_py import _core


def adult_weight(
    bw: np.ndarray,
    ht: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    EIchange: np.ndarray = None,
    NAchange: np.ndarray = None,
    days: int = 365,
    dt: float = 1.0,
    **kwargs: dict,
):
    # Helper to normalize inputs
    bw = np.atleast_1d(bw).astype(float)
    ht = np.atleast_1d(ht).astype(float)
    age = np.atleast_1d(age).astype(float)
    sex_vals = np.array([1.0 if s == "female" else 0.0 for s in np.atleast_1d(sex)])

    # Handle Matrix Inputs
    # R logic: If vector provided, treat as row.
    # C++ logic: expects (n_ind, steps)
    steps = int(np.ceil(days / dt))
    n_ind = len(bw)

    if EIchange is None:
        EIchange = np.zeros((n_ind, steps))
    elif EIchange.ndim == 1:
        # If passed as 1D array (like rep(-250, 365)), reshape to (1, 365)
        if len(EIchange) == steps:
            EIchange = np.tile(EIchange, (n_ind, 1))
        else:
            # Fallback or error handling
            EIchange = EIchange.reshape(n_ind, -1)

    if NAchange is None:
        NAchange = np.zeros((n_ind, steps))
    elif NAchange.ndim == 1:
        if len(NAchange) == steps:
            NAchange = np.tile(NAchange, (n_ind, 1))

    # Note: The C++ shim expects Eigen Matrix. pybind11 converts numpy 2D array -> Eigen Matrix.
    # However, Eigen is Column-Major by default, Numpy is Row-Major.
    # pybind11 handles this, but we pass the array directly.
    # The original R code transposed: EIchange <- t(EIchange).
    # The C++ accesses it as EIchange(time, individual_index).
    # Wait, looking at C++: EIchange(floor(t/dt), _)
    # This implies ROWS are time steps and COLS are individuals?
    # Let's check adult_weight.R: "Each row should represent consumption at each day"
    # So shape should be (Days, Individuals).

    # Our numpy setup above (n_ind, steps) is (Individuals, Days).
    # So we must transpose before passing to C++.

    ei_cpp = EIchange.T  # Shape becomes (Steps, Individuals)
    na_cpp = NAchange.T

    # Defaults for others
    pal = np.full(n_ind, kwargs.get("PAL", 1.5))
    pcarb = np.full(n_ind, kwargs.get("pcarb", 0.5))
    pcarb_base = np.full(n_ind, kwargs.get("pcarb_base", 0.5))

    # Dispatch
    if "EI" in kwargs and "fat" in kwargs:
        return _core.adult_weight_wrapper_EI_fat(
            bw,
            ht,
            age,
            sex_vals,
            ei_cpp,
            na_cpp,
            pal,
            pcarb_base,
            pcarb,
            dt,
            np.atleast_1d(kwargs["EI"]),
            np.atleast_1d(kwargs["fat"]),
            days,
            True,
        )
    else:
        return _core.adult_weight_wrapper(
            bw,
            ht,
            age,
            sex_vals,
            ei_cpp,
            na_cpp,
            pal,
            pcarb_base,
            pcarb,
            dt,
            days,
            True,
        )
