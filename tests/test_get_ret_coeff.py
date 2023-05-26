import numpy as np
from numpy.testing import assert_array_almost_equal

from mwrpy.level2.get_ret_coeff import get_mvr_coeff

FREQ = np.array(
    [
        22.24,
        23.04,
        23.84,
        25.44,
        26.24,
        27.84,
        31.4,
        51.26,
        52.28,
        53.86,
        54.94,
        56.66,
        57.3,
        58.0,
    ]
)
SITE = "hyytiala"


class TestLWP:
    coeff: dict

    def test_lwp_coefficients(self):
        data = get_mvr_coeff(SITE, "lwp", FREQ)
        self.coeff = data[0]
        for key, item in self.coeff.items():
            if isinstance(item, str):
                continue
            match key:
                case (
                    "RP"
                    | "RB"
                    | "CC"
                    | "TS"
                    | "HS"
                    | "ZS"
                    | "IR"
                    | "I1"
                    | "I2"
                    | "SU"
                    | "AL"
                ):
                    data = (0, 0, 0)
                case "DY" | "PS" | "DB" | "RB":
                    data = (1, 1, 1)
                case "RT":
                    data = (2, 2, 2)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (9, 4, 6.5)
                case "FR":
                    data = (23.04, 58, 41.699997)
                case "AG":
                    data = (90, 4.2, 27.473684)
                case "NP":
                    data = (0.025999999, 0.025999999, 0.025999999)
                case "input_offset":
                    data = (61.404, -5.174205e-05, 6337.0576)
                case "output_offset":
                    data = ([1.25075], [17.077847], 5.906157)
                case "output_scale":
                    data = ([1.6676667], [22.770462], 7.8748765)
                case "RM":
                    data = ([0.00434408], [0.19351903], 0.041093733)
                case "FR_BL":
                    data = (23.04, 58.0, 41.699997)
                case "input_scale":
                    data = (0.01880088, 1.0000551, 0.14415415)
                case "W2":
                    data = (-10.072118, -9.450926, -2.7468667)
                case "W1":
                    data = (-107.37779, -2.3326364, -0.7625811)
                case _:
                    self._print_test_data(key)
                    raise ValueError(f"Unknown key: {key}")

            self._check(key, *data)

    def test_iwv_coeffecients(self):
        data = get_mvr_coeff(SITE, "iwv", FREQ)
        self.coeff = data[0]
        for key, item in self.coeff.items():
            if isinstance(item, str):
                continue
            match key:
                case (
                    "RB" | "CC" | "TS" | "HS" | "ZS" | "IR" | "I1" | "I2" | "SU" | "AL"
                ):
                    data = (0, 0, 0)
                case "RP" | "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RT":
                    data = (2, 2, 2)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (9, 4, 6.5)
                case "FR":
                    data = (23.04, 58, 41.699997)
                case "AG":
                    data = (90, 4.2, 27.473684)
                case "NP":
                    data = (0.025999999, 0.025999999, 0.025999999)
                case "input_offset":
                    data = (61.404, -5.174205e-05, 6337.0576)
                case "input_scale":
                    data = (0.01880088, 1.0000551, 0.14415415)
                case "output_offset":
                    data = (24.4429, 333.74542, 115.42163)
                case "output_scale":
                    data = (31.616266, 431.6912, 149.29494)
                case "W1":
                    data = (-4.5245423, -0.9779363, -0.6337075)
                case "W2":
                    data = (-8.668256, 10.204778, -3.7813265)
                case "RM":
                    data = (0.0885116308927536, 3.4751670360565186, 0.73528653383255)
                case "FR_BL":
                    data = (23.04, 58.0, 41.699997)
                case _:
                    self._print_test_data(key)
                    raise ValueError(f"Unknown key: {key}")

            self._check(key, *data)

    def test_tpt_coefficients(self):
        data = get_mvr_coeff(SITE, "tpt", FREQ)
        self.coeff = data[0]
        for key, item in self.coeff.items():
            if isinstance(item, str):
                continue
            match key:
                case ("RB" | "CC" | "TS" | "HS" | "ZS" | "IR" | "I1" | "I2" | "SU"):
                    data = (0, 0, 0)
                case "AL":
                    data = (0.0, 10000.0, 2875.806396484375)
                case "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RP":
                    data = (4, 4, 4)
                case "RT":
                    data = (2, 2, 2)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (12, 4, 8)
                case "FR":
                    data = (23.04, 58, 41.699997)
                case "AG":
                    data = (90, 90, 90)
                case "NP":
                    data = (0.025999999, 0.025999999, 0.025999999)
                case "input_offset":
                    data = (61.40399932861328, -5.174204852664843e-05, 6304.67041015625)
                case "input_scale":
                    data = (
                        0.018800880759954453,
                        1.0000550746917725,
                        0.14665262401103973,
                    )
                case "output_offset":
                    data = (271.20001220703125, 219.82550048828125, 259.2635803222656)
                case "output_scale":
                    data = (44.46666717529297, 24.96733283996582, 32.562801361083984)
                case "W1":
                    data = (-60.52742004394531, 6.198000431060791, -2.250164031982422)
                case "W2":
                    data = (6.222752571105957, 1.4310721158981323, -1.8142919540405273)
                case "RM":
                    data = (2.035391330718994, 3.3711481, 1.2820654379647203)
                case "FR_BL":
                    data = (23.04, 58.0, 41.699997)
                case _:
                    self._print_test_data(key)
                    raise ValueError(f"Unknown key: {key}")

            self._check(key, *data)

    def test_tpb_coefficients(self):
        data = get_mvr_coeff(SITE, "tpb", FREQ)
        self.coeff = data[0]
        for key, item in self.coeff.items():
            if isinstance(item, str):
                continue
            match key:
                case ("RB" | "CC" | "TS" | "HS" | "ZS" | "IR" | "I1" | "I2" | "SU"):
                    data = (0, 0, 0)
                case "AL":
                    data = (0.0, 10000.0, 2875.806396484375)
                case "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RP":
                    data = (5, 5, 5)
                case "RT":
                    data = (2, 2, 2)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (15, 4, 9.5)
                case "FR":
                    data = (23.04, 58, 41.699997)
                case "AG":
                    data = (90.0, 4.199999809265137, 19.440000534057617)
                case "NP":
                    data = (0.025999999, 0.025999999, 0.025999999)
                case "input_offset":
                    data = (
                        61.40399932861328,
                        -5.174204852664843e-05,
                        945.8815307617188,
                    )
                case "input_scale":
                    data = (
                        0.018800880759954453,
                        1.0000550746917725,
                        0.02581867016851902,
                    )
                case "output_offset":
                    data = (271.20001220703125, 219.82550048828125, 259.2635803222656)
                case "output_scale":
                    data = (44.46666717529297, 24.96733283996582, 32.562801361083984)
                case "W1":
                    data = (
                        -149.7417449951172,
                        0.19153261184692383,
                        -0.07408872246742249,
                    )
                case "W2":
                    data = (4.332581996917725, -0.7212945818901062, -1.5545871257781982)
                case "RM":
                    data = (0.8922356963157654, 3.0863358, 1.051631445229202)
                case "FR_BL":
                    data = (23.04, 58.0, 41.699997)
                case _:
                    self._print_test_data(key)
                    raise ValueError(f"Unknown key: {key}")

            self._check(key, *data)

    def test_hpt_coefficients(self):
        data = get_mvr_coeff(SITE, "hpt", FREQ)
        self.coeff = data[0]
        for key, item in self.coeff.items():
            if isinstance(item, str):
                continue
            match key:
                case ("RB" | "CC" | "TS" | "HS" | "ZS" | "IR" | "I1" | "I2" | "SU"):
                    data = (0, 0, 0)
                case "AL":
                    data = (0.0, 10000.0, 2875.806396484375)
                case "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RP":
                    data = (3, 3, 3)
                case "RT":
                    data = (2, 2, 2)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (9, 4, 6.5)
                case "FR":
                    data = (23.04, 58, 41.699997)
                case "AG":
                    data = (90.0, 90, 90)
                case "NP":
                    data = (0.025999999, 0.025999999, 0.025999999)
                case "input_offset":
                    data = (61.40399932861328, -5.174204852664843e-05, 6304.67041015625)
                case "input_scale":
                    data = (
                        0.018800880759954453,
                        1.0000550746917725,
                        0.14665262401103973,
                    )
                case "output_offset":
                    data = (9.317116737365723, 0.07553045451641083, 4.96573543548584)
                case "output_scale":
                    data = (12.174510955810547, 0.10056871920824051, 6.55507755279541)
                case "W1":
                    data = (-22.016393661499023, -2.077390670776367, -4.223663806915283)
                case "W2":
                    data = (-6.617764472961426, -5.968339443206787, -4.423099517822266)
                case "RM":
                    data = (0.7835157513618469, 0.0067931315, 0.44078083002109514)
                case "FR_BL":
                    data = (23.04, 58.0, 41.699997)
                case _:
                    self._print_test_data(key)
                    raise ValueError(f"Unknown key: {key}")

            self._check(key, *data)

    def _check(
        self, key: str, first: float = 0, last: float = 0, mean_value: float = 0
    ):
        item = self.coeff[key]
        assert item.ndim in (1, 2, 3)
        if item.ndim == 1:
            first_value = item[0]
            last_value = item[-1]
        elif item.ndim == 2:
            first_value = item[0, 0]
            last_value = item[-1, -1]
        else:
            first_value = item[0, 0, 0]
            last_value = item[-1, -1, -1]

        # print(key, f"{first_value}, {last_value}, {np.mean(item)}")

        assert_array_almost_equal(first_value, first, decimal=4)
        assert_array_almost_equal(last_value, last, decimal=4)
        assert_array_almost_equal(np.mean(item), mean_value, decimal=4)

    def _print_test_data(self, key: str):
        item = self.coeff[key]
        if item.ndim == 1:
            print(key, f"{item[0]}, {item[-1]}, {np.mean(item)}")
        elif item.ndim == 2:
            print(key, f"{item[0, 0]}, {item[-1, -1]}, {np.mean(item)}")
        else:
            print(key, f"{item[0, 0, 0]}, {item[-1, -1, -1]}, {np.mean(item)}")
