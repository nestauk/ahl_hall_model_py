import numpy as np
from . import _core

# Expose the set_seed function directly
set_seed = _core.set_seed


def adult_weight(bw, ht, age, sex, EIchange=None, NAchange=None, days=365, dt=1.0, **kwargs):
    bw = np.atleast_1d(bw).astype(float)
    ht = np.atleast_1d(ht).astype(float)
    age = np.atleast_1d(age).astype(float)
    sex_input = np.atleast_1d(sex)
    sex_vals = np.array([1.0 if s == "female" else 0.0 for s in sex_input], dtype=float)
    n_ind = len(bw)
    steps = int(np.ceil(days / dt))

    if EIchange is None:
        EIchange = np.zeros((n_ind, steps))
    else:
        EIchange = np.array(EIchange, dtype=float)
        if EIchange.ndim == 1:
            if len(EIchange) == steps:
                EIchange = np.tile(EIchange, (n_ind, 1))
            else:
                EIchange = EIchange.reshape(n_ind, -1)

    if NAchange is None:
        NAchange = np.zeros((n_ind, steps))
    else:
        NAchange = np.array(NAchange, dtype=float)
        if NAchange.ndim == 1:
            if len(NAchange) == steps:
                NAchange = np.tile(NAchange, (n_ind, 1))
            else:
                NAchange = NAchange.reshape(n_ind, -1)

    EI_cpp = EIchange.T.tolist()
    NA_cpp = NAchange.T.tolist()
    PAL = np.full(n_ind, kwargs.get("PAL", 1.5)).astype(float)
    pcarb = np.full(n_ind, kwargs.get("pcarb", 0.5)).astype(float)
    pcarb_base = np.full(n_ind, kwargs.get("pcarb_base", 0.5)).astype(float)

    if "EI" in kwargs and "fat" in kwargs:
        return _core.adult_weight_wrapper_EI_fat(
            bw.tolist(),
            ht.tolist(),
            age.tolist(),
            sex_vals.tolist(),
            EI_cpp,
            NA_cpp,
            PAL.tolist(),
            pcarb_base.tolist(),
            pcarb.tolist(),
            dt,
            np.atleast_1d(kwargs["EI"]).astype(float).tolist(),
            np.atleast_1d(kwargs["fat"]).astype(float).tolist(),
            days,
            True,
        )
    elif "EI" in kwargs or "fat" in kwargs:
        is_energy = "EI" in kwargs
        extradata = np.atleast_1d(kwargs["EI"]) if is_energy else np.atleast_1d(kwargs["fat"])
        return _core.adult_weight_wrapper_EI(
            bw.tolist(),
            ht.tolist(),
            age.tolist(),
            sex_vals.tolist(),
            EI_cpp,
            NA_cpp,
            PAL.tolist(),
            pcarb_base.tolist(),
            pcarb.tolist(),
            dt,
            extradata.astype(float).tolist(),
            days,
            True,
            is_energy,
        )
    else:
        return _core.adult_weight_wrapper(
            bw.tolist(),
            ht.tolist(),
            age.tolist(),
            sex_vals.tolist(),
            EI_cpp,
            NA_cpp,
            PAL.tolist(),
            pcarb_base.tolist(),
            pcarb.tolist(),
            dt,
            days,
            True,
        )


def energy_build(energy, time, interpolation="Brownian"):
    """
    Interpolates energy consumption moments.
    """
    energy = np.array(energy, dtype=float)
    if energy.ndim == 1:
        energy = energy.reshape(1, -1)
    time = np.array(time, dtype=float)

    if energy.shape[1] != len(time):
        raise ValueError(f"Columns ({energy.shape[1]}) != Time ({len(time)})")
    if time[0] != 0:
        raise ValueError("First time element must be 0")
    if np.any(time < 0):
        raise ValueError("Time values must be positive")

    res = _core.EnergyBuilder(energy.tolist(), time.tolist(), interpolation)
    return res[:, 1:]
