import numpy as np
from . import _core


def adult_weight(
    bw, ht, age, sex, EIchange=None, NAchange=None, days=365, dt=1.0, **kwargs
):
    # 1. Normalize inputs to numpy arrays of floats
    bw = np.atleast_1d(bw).astype(float)
    ht = np.atleast_1d(ht).astype(float)
    age = np.atleast_1d(age).astype(float)

    # 2. Handle sex: convert "female"->1.0, "male"->0.0
    sex_input = np.atleast_1d(sex)
    sex_vals = np.array([1.0 if s == "female" else 0.0 for s in sex_input], dtype=float)

    n_ind = len(bw)
    steps = int(np.ceil(days / dt))

    # 3. Prepare Matrices (EIchange / NAchange)
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

    # 4. Transpose and Convert Matrices to LIST OF LISTS
    # This is crucial: C++ expects std::vector<std::vector<double>>
    EI_cpp = EIchange.T.tolist()
    NA_cpp = NAchange.T.tolist()

    # 5. Prepare Optional Parameters
    PAL = np.full(n_ind, kwargs.get("PAL", 1.5)).astype(float)
    pcarb = np.full(n_ind, kwargs.get("pcarb", 0.5)).astype(float)
    pcarb_base = np.full(n_ind, kwargs.get("pcarb_base", 0.5)).astype(float)

    # 6. Dispatch to C++
    # CRITICAL: We must use .tolist() on ALL vector arguments to convert NumPy arrays
    # to Python lists, as standard C++ vectors don't accept NumPy arrays directly.

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
        extradata = (
            np.atleast_1d(kwargs["EI"]) if is_energy else np.atleast_1d(kwargs["fat"])
        )
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
