"""Module for LWP offset correction"""
# import os
# from pathlib import Path
# from time import gmtime

import numpy as np
import pandas as pd

from mwrpy.level1.write_lev1_nc import find_lwcl_free

Fill_Value_Float = -999.0


def correct_lwp_offset(
    lev1: dict, lwp_org: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """This function corrects Lwp offset using the
    2min standard deviation of the 31.4 GHz channel and IR temperature

    Args:
        lev1: Level 1 data.
        lwp: Lwp array.
        index: Index to use.
        site: site: Name of site.

    Examples:
        >>> from level2.lwp_offset import correct_lwp_offset
        >>> correct_lwp_offset(lev1, lwp, index, 'site_name')
    """

    if "elevation_angle" in lev1:
        elevation_angle = lev1["elevation_angle"][:]
    else:
        elevation_angle = 90 - lev1["zenith_angle"][:]

    lwcl_i, _ = find_lwcl_free(lev1, index)
    lwp = np.copy(lwp_org)
    lwp[(lwcl_i != 0) | (lwp > 0.04) | (elevation_angle[index] < 89.0)] = np.nan
    time = lev1["time"][index]
    if max(time) > 25:
        lwp_df = pd.DataFrame({"Lwp": lwp}, index=pd.to_datetime(time, unit="s"))
    else:
        lwp_df = pd.DataFrame({"Lwp": lwp}, index=pd.to_datetime(time, unit="h"))
    lwp_std = lwp_df.rolling("2min", center=True, min_periods=10).std()
    lwp_max = lwp_std.rolling("20min", center=True, min_periods=100).max()
    lwp_df[lwp_max > 0.002] = np.nan
    lwp_offset = lwp_df.rolling("20min", center=True, min_periods=100).mean()

    # use previously determined offset (within 2h) and write current offset in csv file
    #
    # dir_name = str(Path(__file__).absolute().parent.parent)
    #
    # t1 = gmtime(time.data[0])
    # fname = dir_name + "/site_config/" + site + "/lwp_offset_" + str(t1[0]) + ".csv"
    #
    # if not os.path.isfile(fname):
    #     df = pd.DataFrame({"date": [], "offset": []})
    #     df.to_csv(fname)
    #
    # csv_off = pd.read_csv(fname, usecols=["date", "offset"])
    #
    # ind = np.where(lwp_offset["Lwp"].values > 0)[0]
    # if ind.size > 1:
    #     csv_off = pd.concat(
    #         [
    #             csv_off,
    #             pd.DataFrame(
    #                 {"date": time[ind[0]], "offset": lwp_offset["Lwp"][ind[0]]},
    #                 index=[0],
    #             ),
    #         ],
    #         ignore_index=True,
    #     )
    #     csv_off = pd.concat(
    #         [
    #             csv_off,
    #             pd.DataFrame(
    #                 {"date": time[ind[-1]], "offset": lwp_offset["Lwp"][ind[-1]]},
    #                 index=[0],
    #             ),
    #         ],
    #         ignore_index=True,
    #     )
    # elif ind.size == 1:
    #     csv_off = pd.concat(
    #         [
    #             csv_off,
    #             pd.DataFrame(
    #                 {"date": time[ind], "offset": lwp_offset["Lwp"][int(ind)]},
    #                 index=[0],
    #             ),
    #         ],
    #         ignore_index=True,
    #     )
    # csv_off = csv_off.sort_values(by=["date"])
    # csv_off = csv_off.drop_duplicates(subset=["date"])
    # csv_off.to_csv(fname, index=False)
    #
    # # offset from previous day
    # off_ind = np.where(
    #     (csv_off["date"].values < time[0])
    #     & (time[0] - csv_off["date"].values < 2.0 * 3600.0)
    # )[0]
    # if off_ind.size == 1:
    #     off_ind = np.array([int(off_ind), int(off_ind)])
    # if (off_ind.size > 1) & (np.isnan(lwp_offset["Lwp"][0])):
    #     lwp_offset["Lwp"][0] = csv_off["offset"][off_ind[-1]]
    # # offset from next day (for reprocessing purposes)
    # off_ind = np.where(
    #     (csv_off["date"].values > time[-1])
    #     & (csv_off["date"].values - time[-1] < 2.0 * 3600.0)
    # )[0]
    # if off_ind.size == 1:
    #     off_ind = np.array([int(off_ind), int(off_ind)])
    # if (off_ind.size > 1) & (np.isnan(lwp_offset["Lwp"][-1])):
    #     lwp_offset["Lwp"][-1] = csv_off["offset"][off_ind[0]]

    lwp_offset = lwp_offset.interpolate(method="linear")
    lwp_offset = lwp_offset.fillna(method="bfill")
    lwp_offset["Lwp"][np.isnan(lwp_offset["Lwp"])] = 0
    lwp_org -= lwp_offset["Lwp"].values

    return lwp_org, lwp_offset["Lwp"].values
