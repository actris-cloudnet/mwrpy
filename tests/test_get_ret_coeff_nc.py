import numpy as np
from numpy.testing import assert_array_almost_equal

from mwrpy.level2.get_ret_coeff import get_mvr_coeff

FREQ = np.array(
    [
        22.23999977111816406250,
        23.04000091552734375000,
        23.84000015258789062500,
        25.44000053405761718750,
        26.23999977111816406250,
        27.84000015258789062500,
        31.39999961853027343750,
        51.25999832153320312500,
        52.27999877929687500000,
        53.86000061035156250000,
        54.93999862670898437500,
        56.65999984741210937500,
        57.29999923706054687500,
        58.00000000000000000000,
    ]
)
SITE = "juelich"


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
                    data = (-99, -99, -99)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (9, 4, 6.5)
                case "FR":
                    data = (22.239999771118164, -999.0, -486.6399999346052)
                case "AG":
                    data = (90, 90, 90)
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
                case "TL":
                    data = (0.0012005792232230306, 0.0, 0.0008158495267187911)
                case "TQ":
                    data = (-1.0344068869017065e-05, 0.0, 9.621058081523057e-06)
                case "OS":
                    data = (
                        -0.15209339559078217,
                        -0.15209339559078217,
                        -0.15209339559078217,
                    )
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
                    data = (-99, -99, -99)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (9, 4, 6.5)
                case "FR":
                    data = (22.239999771118164, -999.0, -486.6399999346052)
                case "AG":
                    data = (90, 90, 90)
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
                case "TL":
                    data = (0.11296015977859497, 0.0, 0.02241739258170128)
                case "TQ":
                    data = (-0.0005557882832363248, 0.0, 9.281754942743905e-05)
                case "OS":
                    data = (
                        -0.5454273819923401,
                        -0.5454273819923401,
                        -0.5454273819923401,
                    )
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
                    data = (0.0, 10000.0, 3181.976806640625)
                case "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RP":
                    data = (4, 4, 4)
                case "RT":
                    data = (-99, -99, -99)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (12, 4, 8)
                case "FR":
                    data = (-999.0, 58.0, -472.05000032697404)
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
                case "TL":
                    data = (0.0, 3.808150053024292, 0.06220490690258848)
                case "TQ":
                    data = (0.0, -0.006185657810419798, 2.8331756044262935e-05)
                case "OS":
                    data = (-119.42852020263672, 1605.0447998046875, -8.820448498393214)
                case "n_height_grid":
                    data = (43, 43, 43)
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
                    data = (0.0, 10000.0, 3181.976806640625)
                case "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RP":
                    data = (5, 5, 5)
                case "RT":
                    data = (-99, -99, -99)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (15, 4, 9.5)
                case "FR":
                    data = (51.2599983215332, 58.0, 54.89999771118164)
                case "AG":
                    data = (5.400000095367432, 90.0, 32.79999923706055)
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
                    data = (54.939998626708984, 58.0, 56.724998474121094)
                case "n_height_grid":
                    data = (43, 43, 43)
                case "OS":
                    data = (-2.1868269443511963, 80.95862579345703, 4.412421703338623)
                case "TL":
                    data = (
                        0.021718325093388557,
                        -0.13281039893627167,
                        0.03465010225772858,
                    )
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
                    data = (0.0, 10000.0, 3181.976806640625)
                case "DB" | "PS" | "DY":
                    data = (1, 1, 1)
                case "RP":
                    data = (3, 3, 3)
                case "RT":
                    data = (-99, -99, -99)
                case "VN":
                    data = (110, 110, 110)
                case "ND":
                    data = (9, 4, 6.5)
                case "FR":
                    data = (22.239999771118164, -999.0, -486.6399999346052)
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
                case "TL":
                    data = (-0.00020702094479929656, 0.0, 5.805333452826366e-06)
                case "TQ":
                    data = (1.1459743518571486e-06, 0.0, 2.6153051095753103e-08)
                case "OS":
                    data = (
                        0.001587925013154745,
                        6.240158199943835e-06,
                        9.005275249068672e-05,
                    )
                case "n_height_grid":
                    data = (43, 43, 43)
                case _:
                    self._print_test_data(key)
                    raise ValueError(f"Unknown key: {key}")

            self._check(key, *data)

    def _check(
        self, key: str, first: float = 0, last: float = 0, mean_value: float = 0
    ):
        item = np.array(self.coeff[key])

        assert item.ndim in (0, 1, 2, 3)

        if item.ndim == 0:
            first_value = item
            last_value = item
        elif item.ndim == 1:
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
        item = np.array(self.coeff[key])
        if item.ndim == 0:
            print(key, f"{item}, {item}, {item}")
        elif item.ndim == 1:
            print(key, f"{item[0]}, {item[-1]}, {np.mean(item)}")
        elif item.ndim == 2:
            print(key, f"{item[0, 0]}, {item[-1, -1]}, {np.mean(item)}")
        else:
            print(key, f"{item[0, 0, 0]}, {item[-1, -1, -1]}, {np.mean(item)}")
