"""Module to load in retrieval coefficient files"""
import netCDF4 as nc
import numpy as np

from mwrpy.utils import get_coeff_list

Fill_Value_Float = -999.0
Fill_Value_Int = -99


def get_mvr_coeff(
    site: str | None, prefix: str, freq: np.ndarray, coeff_files: list | None
):
    """This function extracts retrieval coefficients for given files.

    Args:
        coeff_files: List of coefficient files.
        site: Name of site.
        prefix: Identifier for type of product.
        freq: Frequencies of observations.

    Examples:
        >>> from mwrpy.level2.get_ret_coeff import get_mvr_coeff
        >>> get_mvr_coeff('site_name', 'lwp', np.array([22, 31.4]))
    """

    c_list = get_coeff_list(site, prefix, coeff_files)

    coeff: dict = {}

    if (str(c_list[0][-3:]).lower() == "ret") and (len(c_list) == 1):
        coeff = read_coeff_ascii(c_list[0])
        if "AL" not in coeff:
            coeff["AL"] = [0]
        if prefix == "tpb":
            for key in ("W1", "W2"):
                coeff[key] = coeff[key].squeeze(axis=2)
            for key in ("input_offset", "input_scale", "output_offset", "output_scale"):
                coeff[key] = coeff[key].squeeze(axis=0)

        aux = [
            "TS",
            "HS",
            "PS",
            "ZS",
            "IR",
            "I1",
            "I2",
            "DY",
            "SU",
        ]
        for aux_i in aux:
            if aux_i not in coeff:
                coeff[aux_i] = 0
        coeff["FR_BL"] = coeff["FR"]

    elif (str(c_list[0][-2:]).lower() == "nc") and (len(c_list) > 0):
        coeff["RT"] = Fill_Value_Int
        N = len(c_list)

        if prefix in ("lwp", "iwv"):
            coeff["AG"] = np.ones(N) * Fill_Value_Float
            coeff["FR"] = np.ones([len(freq), N]) * Fill_Value_Float
            coeff["TL"] = np.zeros([N, len(freq)])
            coeff["TQ"] = np.zeros([N, len(freq)])
            coeff["OS"] = np.zeros(N)
            coeff["AL"] = [0]

            for i_file, file in enumerate(c_list):
                c_file = nc.Dataset(file)
                coeff["AG"][i_file] = c_file["elevation_predictor"][i_file]
                _, freq_ind, freq_coeff = np.intersect1d(
                    freq[:], c_file["freq"][:], assume_unique=False, return_indices=True
                )
                if len(freq_coeff) < len(c_file["freq"][:]):
                    raise RuntimeError(
                        ["Instrument and retrieval frequencies do not match."]
                    )

                coeff["FR"][freq_ind, i_file] = c_file["freq"][freq_coeff]
                coeff["TL"][i_file, freq_ind] = c_file["coefficient_mvr"][freq_coeff]
                if c_file.regression_type == "quadratic":
                    coeff["TQ"][i_file, freq_ind] = c_file["coefficient_mvr"][
                        freq_coeff + len(freq_coeff)
                    ]
                coeff["OS"][i_file] = c_file["offset_mvr"][0]

        elif prefix in ("tpt", "hpt"):
            c_file = nc.Dataset(c_list[0])
            n_height_grid = c_file.dimensions["n_height_grid"].size

            coeff["AG"] = np.ones(N) * Fill_Value_Float
            coeff["FR"] = np.ones([len(freq), N]) * Fill_Value_Float
            coeff["TL"] = np.zeros([N, n_height_grid, len(freq)])
            coeff["TQ"] = np.zeros([N, n_height_grid, len(freq)])
            coeff["OS"] = np.zeros([n_height_grid, N])
            coeff["n_height_grid"] = n_height_grid
            coeff["AL"] = c_file["height_grid"]

            for i_file, file in enumerate(c_list):
                c_file = nc.Dataset(file)
                coeff["AG"][i_file] = c_file["elevation_predictor"][i_file]
                _, freq_ind, freq_coeff = np.intersect1d(
                    freq[:], c_file["freq"][:], assume_unique=False, return_indices=True
                )
                if len(freq_coeff) < len(c_file["freq"][:]):
                    raise RuntimeError(
                        ["Instrument and retrieval frequencies do not match."]
                    )

                coeff["FR"][freq_ind, i_file] = c_file["freq"][freq_coeff]
                coeff["TL"][i_file, :, freq_ind] = c_file["coefficient_mvr"][
                    freq_coeff, :
                ]
                if c_file.regression_type == "quadratic":
                    coeff["TQ"][i_file, :, freq_ind] = c_file["coefficient_mvr"][
                        freq_coeff + len(freq_coeff), :
                    ]
                coeff["OS"][:, i_file] = c_file["offset_mvr"][:]

        elif prefix == "tpb":
            c_file = nc.Dataset(c_list[0])
            _, freq_ind, freq_coeff = np.intersect1d(
                freq[:], c_file["freq"][:], assume_unique=False, return_indices=True
            )
            if len(freq_coeff) < len(c_file["freq"][:]):
                raise RuntimeError(
                    ["Instrument and retrieval frequencies do not match."]
                )

            coeff["AG"] = np.sort(c_file["elevation_predictor"][:])
            coeff["AL"] = c_file["height_grid"]
            coeff["FR"] = c_file["freq"]
            coeff["FR_BL"] = c_file["freq_bl"]
            coeff["n_height_grid"] = c_file.dimensions["n_height_grid"].size
            coeff["OS"] = c_file["offset_mvr"][:]
            coeff["TL"] = c_file["coefficient_mvr"][:, :]

        else:
            raise RuntimeError(
                [
                    "Prefix "
                    + prefix
                    + " not recognized for retrieval coefficient file(s)."
                ]
            )

    if (coeff["RT"] < 2) & (len(coeff["AL"]) == 1):

        def f_offset(x):
            return np.array(
                [coeff["OS"][(np.abs(coeff["AG"] - v)).argmin()] for v in x]
            )

        if coeff["RT"] in (0, 1):
            coeff["TL"] = coeff["TL"][np.newaxis, :]

        def f_lin(x):
            return np.array(
                [coeff["TL"][(np.abs(coeff["AG"] - v)).argmin(), :] for v in x]
            )

        if coeff["RT"] in (1, Fill_Value_Int):
            if coeff["RT"] == 1:
                coeff["TQ"] = coeff["TQ"][np.newaxis, :]

            def f_quad(x):
                return np.array(
                    [coeff["TQ"][(np.abs(coeff["AG"] - v)).argmin(), :] for v in x]
                )

    elif (coeff["RT"] < 2) & (len(coeff["AL"]) > 1) & (prefix != "tpb"):

        def f_offset(x):
            return np.array(
                [coeff["OS"][:, (np.abs(coeff["AG"] - v)).argmin()] for v in x]
            )

        if coeff["RT"] in (0, 1):
            coeff["TL"] = coeff["TL"][np.newaxis, :, :]

        def f_lin(x):
            return np.array(
                [coeff["TL"][(np.abs(coeff["AG"] - v)).argmin(), :, :] for v in x]
            )

        if coeff["RT"] in (1, Fill_Value_Int):
            if coeff["RT"] == 1:
                coeff["TQ"] = coeff["TQ"][np.newaxis, :, :]

            def f_quad(x):
                return np.array(
                    [coeff["TQ"][(np.abs(coeff["AG"] - v)).argmin(), :, :] for v in x]
                )

    elif (coeff["RT"] < 2) & (len(coeff["AL"]) > 1) & (prefix == "tpb"):

        def f_offset(_x):
            return coeff["OS"]

        def f_lin(_x):
            return coeff["TL"]

        def f_quad(_x):
            return np.empty(0)

    elif coeff["RT"] == 2:

        def input_scale(x):
            return np.array(
                [coeff["input_scale"][(np.abs(coeff["AG"] - v)).argmin(), :] for v in x]
            )

        def input_offset(x):
            return np.array(
                [
                    coeff["input_offset"][(np.abs(coeff["AG"] - v)).argmin(), :]
                    for v in x
                ]
            )

        def output_scale(x):
            return np.array(
                [
                    coeff["output_scale"][(np.abs(coeff["AG"] - v)).argmin(), :]
                    for v in x
                ]
            )

        def output_offset(x):
            return np.array(
                [
                    coeff["output_offset"][(np.abs(coeff["AG"] - v)).argmin(), :]
                    for v in x
                ]
            )

        def weights1(x):
            return np.array(
                [coeff["W1"][:, :, (np.abs(coeff["AG"] - v)).argmin()] for v in x]
            )

        if len(coeff["AL"]) > 1:

            def weights2(x):
                return np.array(
                    [coeff["W2"][:, :, (np.abs(coeff["AG"] - v)).argmin()] for v in x]
                )

        else:

            def weights2(x):
                return np.array(
                    [coeff["W2"][(np.abs(coeff["AG"] - v)).argmin(), :] for v in x]
                )

        def factor(x):
            return np.array(
                [coeff["NP"][(np.abs(coeff["AG"] - v)).argmin()] for v in x]
            )

    if str(c_list[0][-3:]).lower() == "ret":
        retrieval_type = ["linear regression", "quadratic regression", "neural network"]
        coeff["retrieval_type"] = retrieval_type[int(coeff["RT"][0])]
        coeff["retrieval_elevation_angles"] = str(coeff["AG"])
        coeff["retrieval_frequencies"] = str(coeff["FR"][:])
        if coeff["TS"] == 0:
            coeff["retrieval_auxiliary_input"] = "no_surface"
        else:
            coeff["retrieval_auxiliary_input"] = "surface"
        coeff["retrieval_description"] = "RPG retrieval"

    elif str(c_list[0][-2:]).lower() == "nc":
        coeff["retrieval_type"] = c_file.regression_type
        coeff["retrieval_elevation_angles"] = str(coeff["AG"])
        coeff["retrieval_frequencies"] = str(c_file["freq"][:])
        coeff["retrieval_auxiliary_input"] = c_file.surface_mode
        coeff["retrieval_description"] = c_file.retrieval_version

    return (
        (coeff, f_offset, f_lin, f_quad)
        if (coeff["RT"] < 2)
        else (
            coeff,
            input_scale,
            input_offset,
            output_scale,
            output_offset,
            weights1,
            weights2,
            factor,
        )
    )


def read_coeff_ascii(coeff_file: str) -> dict:
    """Reads the coefficients from an ascii file."""
    with open(coeff_file, "r", encoding="utf8") as f:
        lines = f.readlines()
    coeff = _read_ns(lines)
    for line in lines:
        if "=" in line[:3]:
            key = line[:2]
            if key not in ("NS", "SL", "SQ"):
                coeff[key] = _parse_lines(f"{key}=", lines)
    return coeff


def _read_ns(lines: list) -> dict:
    d: dict = {
        "input_offset": [],
        "input_scale": [],
        "output_offset": [],
        "output_scale": [],
    }
    for lineno, line in enumerate(lines):
        if line.startswith("NS="):
            d["input_offset"].append(_split_line(lines[lineno]))
            d["input_scale"].append(_split_line(lines[lineno + 1]))
            d["output_offset"].append(_split_line(lines[lineno + 2]))
            d["output_scale"].append(_split_line(lines[lineno + 3]))
    return {key: np.array(value).astype(np.float32) for key, value in d.items()}


def _parse_lines(prefix: str, lines: list) -> np.ndarray:
    data = []
    n_rows = 0
    for lineno, line in enumerate(lines):
        if line.startswith(prefix):
            n_rows += 1
            data.append(_split_line(line))
            for next_line in lines[lineno + 1 :]:
                if next_line.startswith(":"):
                    data.append(_split_line(next_line))
                else:
                    break
    return _reshape_array(data, n_rows, prefix)


def _reshape_array(data: list, n_rows: int, prefix: str) -> np.ndarray:
    data_squeezed: list[str] | list[list[str]]
    if len(data) == 1 and isinstance(data[0], list) and len(data[0]) == 1:
        data_squeezed = data[0]
    else:
        data_squeezed = data
    try:
        array = np.array(data_squeezed).astype(np.float32)
    except ValueError:
        array = np.array(data_squeezed).astype(str)
    if len(array) > n_rows and prefix not in ("TL=", "TQ="):
        array = np.reshape(array, (n_rows, -1, array.shape[1]))
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2 and array.shape[0] == 1:
        array = np.squeeze(array)
    if array.ndim == 3 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    return array


def _split_line(line: str) -> list[str]:
    delimiter = ":" if ":" in line else "="
    return line.split(delimiter)[1].split("#")[0].split()
