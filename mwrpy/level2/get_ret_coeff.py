"""Module to load in retrieval coefficient files"""
import netCDF4 as nc
import numpy as np

from mwrpy.utils import get_coeff_list

Fill_Value_Float = -999.0
Fill_Value_Int = -99


def get_mvr_coeff(site: str, prefix: str, freq: np.ndarray):
    """This function extracts retrieval coefficients for given files.

    Args:
        site: Name of site.
        prefix: Identifier for type of product.
        freq: Frequencies of observations.

    Examples:
        >>> from mwrpy.level2.get_ret_coeff import get_mvr_coeff
        >>> get_mvr_coeff('site_name', 'lwp', np.array([22, 31.4]))
    """

    c_list = get_coeff_list(site, prefix)
    coeff: dict = {}

    if (str(c_list[0][-3:]).lower() == "ret") & (len(c_list) == 1):
        with open(c_list[0], "r", encoding="utf8") as f:
            lines = f.readlines()
            lines = [line.rstrip("\n") for line in lines]
            line_count = len(lines)
            line_num = -1
            while line_num < line_count - 1:
                line_num += 1
                line = lines[line_num]
                if ("=" in line) & (line[0] not in ("#", ":")):
                    if "#" in line:
                        line = line.split("#")[0]
                    name, tmp = line.split("=")
                    if name not in ("SL", "SQ"):
                        if not tmp.strip()[0].isalpha():
                            value = np.array(
                                [float(idx) for idx in tmp.split()], np.float32
                            )
                    if name == "NS":
                        name_list = [
                            "input_offset",
                            "input_scale",
                            "output_offset",
                            "output_scale",
                        ]
                        for split_name in name_list:
                            if split_name in coeff:
                                coeff[split_name] = np.vstack(
                                    (coeff[split_name], value)
                                )
                            else:
                                coeff[split_name] = value
                            if split_name != "output_scale":
                                line_num += 1
                                _, tmp = lines[line_num].split(":")
                                value = np.array(
                                    [float(idx) for idx in tmp.split()], np.float32
                                )
                        line_num -= 1
                    elif name == "W1":
                        w1_stack = value
                        while lines[line_num + 1][0:2] != "W2":
                            line_num += 1
                            _, tmp = lines[line_num].split(":")
                            tmp_splitted = tmp.split()
                            value = np.array(
                                [float(tmp_splitted[idx]) for idx in range(len(value))],
                                np.float32,
                            )
                            w1_stack = np.vstack((w1_stack, value))
                        if name in coeff:
                            if coeff[name].ndim == 3:
                                coeff[name] = np.concatenate(
                                    (coeff[name], w1_stack[:, :, np.newaxis]), axis=2
                                )
                            else:
                                coeff[name] = np.stack((coeff[name], w1_stack), axis=2)
                        else:
                            coeff[name] = w1_stack
                    elif name == "W2":
                        w2_stack = value
                        for _ in range(len(coeff["AL"]) - 1):
                            line_num += 1
                            _, tmp = lines[line_num].split(":")
                            tmp_splitted = tmp.split()
                            value = np.array(
                                [float(tmp_splitted[idx]) for idx in range(len(value))],
                                np.float32,
                            )
                            w2_stack = np.vstack((w2_stack, value))
                        if name in coeff:
                            if coeff[name].ndim == 3:
                                coeff[name] = np.concatenate(
                                    (coeff[name], w2_stack[:, :, np.newaxis]), axis=2
                                )
                            elif (coeff[name].ndim == 2) & (w2_stack.ndim == 1):
                                coeff[name] = np.concatenate(
                                    (coeff[name], w2_stack[np.newaxis, :]), axis=0
                                )
                            elif (coeff[name].ndim == 2) & (w2_stack.ndim == 2):
                                coeff[name] = np.stack((coeff[name], w2_stack), axis=2)
                            else:
                                coeff[name] = np.vstack((coeff[name], w2_stack))
                        else:
                            coeff[name] = w2_stack
                    elif name == "RM":
                        rm_stack = value
                        for _ in range(len(coeff["AL"]) - 1):
                            line_num += 1
                            _, tmp = lines[line_num].split(":")
                            tmp_splitted = tmp.split()
                            rm_stack = np.vstack((rm_stack, float(tmp_splitted[0])))
                        if name in coeff:
                            if (coeff[name].ndim > 1) & (rm_stack.ndim > 1):
                                coeff[name] = np.concatenate(
                                    (coeff[name], rm_stack), axis=1
                                )
                            elif (coeff[name].ndim > 1) & (rm_stack.ndim == 1):
                                coeff[name] = np.concatenate(
                                    (coeff[name], rm_stack[np.newaxis, :]), axis=0
                                )
                            else:
                                coeff[name] = np.vstack((coeff[name], rm_stack))
                        else:
                            coeff[name] = rm_stack
                    elif name == "OS":
                        if "AL" not in coeff:
                            coeff["AL"] = [0]
                        os_stack = value
                        for _ in range(len(coeff["AL"]) - 1):
                            line_num += 1
                            _, tmp = lines[line_num].split(":")
                            tmp_splitted = tmp.split()
                            os_stack = np.vstack((os_stack, float(tmp_splitted[0])))
                        if name in coeff:
                            coeff[name] = np.concatenate(
                                (coeff[name], os_stack), axis=1
                            )
                        else:
                            coeff[name] = os_stack
                    elif name == "TL":
                        tl_stack = value
                        for _ in range(len(coeff["AL"]) - 1):
                            line_num += 1
                            _, tmp = lines[line_num].split(":")
                            tmp_splitted = tmp.split()
                            value = np.array(
                                [
                                    float(tmp_splitted[idx])
                                    for idx in range(len(coeff["FR"]))
                                ],
                                np.float32,
                            )
                            tl_stack = np.vstack((tl_stack, value))
                        coeff[name] = tl_stack
                    elif name == "TQ":
                        tq_stack = value
                        for _ in range(len(coeff["AL"]) - 1):
                            line_num += 1
                            _, tmp = lines[line_num].split(":")
                            tmp_splitted = tmp.split()
                            value = np.array(
                                [
                                    float(tmp_splitted[idx])
                                    for idx in range(len(coeff["FR"]))
                                ],
                                np.float32,
                            )
                            tq_stack = np.vstack((tq_stack, value))
                        coeff[name] = tq_stack
                    else:
                        if name in coeff:
                            coeff[name] = np.vstack((coeff[name], value))
                        else:
                            coeff[name] = value

        f.close()

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

    elif (str(c_list[0][-2:]).lower() == "nc") & (len(c_list) > 0):
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
        if len(coeff["AG"]) == 1:
            coeff["W1"] = coeff["W1"][:, :, np.newaxis]
            coeff["W2"] = coeff["W2"][:, :, np.newaxis]
            coeff["input_scale"] = coeff["input_scale"][np.newaxis, :]
            coeff["input_offset"] = coeff["input_offset"][np.newaxis, :]
            coeff["output_scale"] = coeff["output_scale"][np.newaxis, :]
            coeff["output_offset"] = coeff["output_offset"][np.newaxis, :]

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
        coeff["retrieval_type"] = retrieval_type[int(coeff["RT"])]
        coeff["retrieval_elevation_angles"] = str(coeff["AG"])
        coeff["retrieval_frequencies"] = str(coeff["FR"][:])
        if coeff["TS"] == 0:
            coeff["retrieval_auxiliary_input"] = "no_surface"
        else:
            coeff["retrieval_auxiliary_input"] = "surface"
        coeff["retrieval_description"] = "Neural network"

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
