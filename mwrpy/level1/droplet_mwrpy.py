"""Module for liquid layer (free) period detection. Partly adapted from CloudnetPy."""

from os import PathLike

import numpy as np
import pandas as pd
import scipy.signal
from numpy import ma

from mwrpy.utils import n_elements, read_lidar, time_to_datetime_index


def find_liquid(
    obs: dict,
    time_mwr: np.ndarray,
    tb_std: np.ndarray,
    peak_amp: float = 1e-6,
    max_width: float = 300,
    min_points: int = 3,
    min_top_der: float = 1e-7,
    tb_th: float = 0.15,
    min_alt: float = 100,
) -> np.ndarray:
    """Estimate liquid layers from SNR-screened attenuated backscatter.

    Args:
        obs: Dict for lidar data.
        time_mwr: Time array of the MWR.
        tb_std: Array of brightness temperature standard deviation.
        peak_amp: Minimum value of peak. Default is 1e-6.
        max_width: Maximum width of peak. Default is 300 (m).
        min_points: Minimum number of valid points in peak. Default is 3.
        min_top_der: Minimum derivative above peak, defined as
            (beta_peak-beta_top) / (alt_top-alt_peak). Default is 1e-7.
        tb_th: Threshold for brightness temperature standard deviation
        min_alt: Minimum altitude of the peak from the ground. Default is 100 (m).

    Returns:
        2-D boolean array denoting liquid layers.

    References:
        The method is based on Tuononen, M. et al., 2019,
        https://acp.copernicus.org/articles/19/1985/2019/.

    """

    def _is_proper_peak() -> np.ndarray | bool:
        conditions = (
            npoints >= min_points,
            peak_width < max_width,
            top_der > min_top_der,
            is_high_std,
            peak_alt > min_alt,
        )
        return is_high_std50 | all(conditions)

    time = obs["time"]
    height = obs["height"]
    beta = ma.copy(obs["beta"])
    tb_std = interpolate_tb(time, time_mwr, tb_std)

    is_liquid = np.zeros(time.shape, dtype=int)
    base_below_peak = n_elements(height, 200)
    top_above_peak = n_elements(height, 150)
    difference = ma.array(np.diff(beta, axis=1))
    beta_diff = difference.filled(0)
    beta = beta.filled(0)
    peak_indices = _find_strong_peaks(beta, peak_amp)

    for n, peak in zip(*peak_indices, strict=True):
        lprof = beta[n, :]
        dprof = beta_diff[n, :]
        try:
            base = ind_base(dprof, peak, base_below_peak, 4)
            top = ind_top(dprof, peak, height.shape[0], top_above_peak, 4)
        except IndexError:
            continue
        npoints = np.count_nonzero(lprof[base : top + 1])
        peak_width = height[top] - height[base]
        peak_alt = height[peak] - height[0]
        top_der = (lprof[peak] - lprof[top]) / (height[top] - height[peak])
        is_high_std = tb_std[n] >= tb_th
        is_high_std50 = tb_std[n] >= tb_th * 1.5
        if _is_proper_peak():
            is_liquid[n] = 1

    return is_liquid


def ind_base(dprof: np.ndarray, ind_peak: int, dist: int, lim: float) -> int:
    """Finds base index of a peak in profile.

    Return the lowermost index of profile where 1st order differences
    below the peak exceed a threshold value.

    Args:
        dprof: 1-D array of 1st discrete difference. Masked values should
            be 0, e.g. dprof = np.diff(masked_prof).filled(0)
        ind_peak: Index of (possibly local) peak in the original profile.
            Note that the peak must be found with some other method before
            calling this function.
        dist: Number of elements investigated below *p*. If ( *p* - *dist*)<0,
            search starts from index 0.
        lim: Parameter for base index. Values greater than 1.0 are valid.
            Values close to 1 most likely return the point right below the
            maximum 1st order difference (within *dist* points below *p*).
            Values larger than 1 more likely accept some other point, lower
            in the profile.

    Returns:
        Base index of the peak.

    Raises:
        IndexError: Can't find proper base index (probably too many masked
            values in the profile).

    Examples:
        Consider a profile

        >>> x = np.array([0, 0.5, 1, -99, 4, 8, 5])

        that contains one bad, masked value

        >>> mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0, 0, 0])
            [0, 0.5, 1.0, --, 4.0, 8.0, 5.0]

        The 1st order difference is now

        >>> dx = np.ma.diff(mx).filled(0)
            [0.5, 0.5, 0, 0, 4, -3]

        From the original profile we see that the peak index is 5.
        Let's assume our base can't be more than 4 elements below
        peak and the threshold value is 2. Thus, we call

        >>> ind_base(dx, 5, 4, 2)
            4

        When x[4] is the lowermost point that satisfies the condition.
        Changing the threshold value would alter the result

        >>> ind_base(dx, 5, 4, 10)
            1

    See Also:
        droplet.ind_top()

    """
    start = max(ind_peak - dist, 0)  # should not be negative
    diffs = dprof[start:ind_peak]
    mind = np.argmax(diffs)
    return start + np.where(diffs > diffs[mind] / lim)[0][0]


def ind_top(dprof: np.ndarray, ind_peak: int, nprof: int, dist: int, lim: float) -> int:
    """Finds top index of a peak in profile.

    Return the uppermost index of profile where 1st order differences
    above the peak exceed a threshold value.

    Args:
        dprof: 1-D array of 1st discrete difference. Masked values should be 0, e.g.
            dprof = np.diff(masked_prof).filled(0)
        nprof: Length of the profile. Top index can't be higher than this.
        ind_peak: Index of (possibly local) peak in the profile. Note that the peak
            must be found with some other method before calling this function.
        dist: Number of elements investigated above *p*. If (*p* + *dist*) > *nprof*,
            search ends to *nprof*.
        lim: Parameter for top index. Values greater than 1.0 are valid. Values close
            to 1 most likely return the point right above the maximum 1st order
            difference (within *dist* points above *p*). Values larger than 1 more
            likely accept some other point, higher in the profile.

    Returns:
        Top index of the peak.

    Raises:
        IndexError: Can not find proper top index (probably too many masked
            values in the profile).

    See Also:
        droplet.ind_base()

    """
    end = min(ind_peak + dist, nprof)  # should not be greater than len(profile)
    diffs = dprof[ind_peak:end]
    mind = np.argmin(diffs)
    return ind_peak + np.where(diffs < diffs[mind] / lim)[0][-1] + 1


def interpolate_tb(
    time: np.ndarray, time_mwr: np.ndarray, tb: np.ndarray
) -> np.ndarray:
    """Linear interpolation of brightness temperature standard deviation to fill masked values.

    Args:
        time: Time array.
        time_mwr: Time array of the MWR.
        tb: Array of brightness temperature standard deviation.

    Returns:
        Brightness temperature standard deviation where the masked values are filled by interpolation.

    """
    if tb.all() is ma.masked:
        return np.zeros(time.shape)
    ind = ma.where(tb)[0]
    return np.interp(time, time_mwr[ind], tb[ind])


def _find_strong_peaks(data: np.ndarray, threshold: float) -> tuple:
    """Finds local maximums from data (greater than *threshold*)."""
    peaks = scipy.signal.argrelextrema(data, np.greater, order=4, axis=1)
    strong_peaks = np.where(data[peaks] > threshold)
    return peaks[0][strong_peaks], peaks[1][strong_peaks]


def find_lwcl_free(
    lev1: dict, path_to_lidar: str | PathLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """Identifying liquid water cloud free periods using 31.4 GHz TB variability.
    Uses water vapor channel as proxy for a humidity dependent threshold.
    """
    index = np.ones(len(lev1["time"]), dtype=np.int32)
    status = np.zeros(len(lev1["time"]), dtype=np.int32)

    # Different frequencies for window and water vapor channels depending on instrument type
    freq_win = np.where(
        (np.isclose(np.round(lev1["frequency"][:], 1), 31.4))
        | (np.isclose(np.round(lev1["frequency"][:], 1), 190.8))
    )[0]
    freq_win = np.array([freq_win[0]]) if len(freq_win) > 1 else freq_win
    freq_wv = np.where(
        (np.isclose(np.round(lev1["frequency"][:], 1), 22.2))
        | (np.isclose(np.round(lev1["frequency"][:], 1), 183.9))
    )[0]
    if len(freq_win) == 1 and len(freq_wv) == 1:
        tb = np.squeeze(lev1["tb"][:, freq_win])
        tb[(lev1["pointing_flag"][:] == 1) | (lev1["elevation_angle"][:] < 89.0)] = (
            np.nan
        )
        ind = time_to_datetime_index(lev1["time"][:])
        tb_df = pd.DataFrame({"Tb": tb}, index=ind)
        offset = "3min" if np.nanmean(np.diff(lev1["time"])) < 1.8 else "10min"
        tb_std = tb_df.rolling(
            pd.tseries.frequencies.to_offset(offset), center=True, min_periods=50
        ).std()
        offset = "20min" if np.nanmean(np.diff(lev1["time"])) < 1.8 else "60min"
        tb_mx = tb_std.rolling(
            pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
        ).max()

        tb_wv = np.squeeze(lev1["tb"][:, freq_wv])
        tb_rat = pd.DataFrame({"Tb": tb_wv / tb}, index=ind)
        tb_rat = tb_rat.rolling(
            pd.tseries.frequencies.to_offset(offset), center=True, min_periods=100
        ).max()

        index_rem = np.array(range(len(lev1["time"])))
        if path_to_lidar:
            # Use lidar data (Cloudnet format) to identify liquid water clouds
            lidar = read_lidar(path_to_lidar)
            mwr_ind = [
                i
                for i, tt in enumerate(lev1["time"])
                if np.min(np.abs(tt - lidar["time"])) < 600
            ]
            lidar_ind = [
                i
                for i, tt in enumerate(lidar["time"])
                if np.min(np.abs(tt - lev1["time"])) < 600
            ]
            if len(mwr_ind) > 0:
                fact = (
                    0.75
                    if np.isclose(np.round(lev1["frequency"][freq_win], 1), 190.8)
                    else 0.1
                )
                liquid_from_lidar = find_liquid(
                    lidar,
                    lev1["time"][mwr_ind],
                    tb_mx["Tb"].iloc[mwr_ind].values,
                    tb_th=float(np.nanmedian(tb_rat["Tb"]) * fact),
                )
                liquid_flag = pd.DataFrame(
                    {"lf": liquid_from_lidar[lidar_ind]},
                    index=time_to_datetime_index(lidar["time"][lidar_ind]),
                )
                liquid_flag = liquid_flag.resample(
                    "20min", origin="start", closed="left", label="left", offset="10min"
                ).max()
                liquid_flag = liquid_flag.reindex(
                    tb_df.index[mwr_ind], method="nearest"
                )
                liquid_flag = liquid_flag.fillna(value=2.0)
                index[mwr_ind] = np.array(liquid_flag["lf"][:].values, dtype=np.int32)
                status[mwr_ind] = 1
                index_rem = np.setxor1d(index_rem, mwr_ind)

        index[
            index_rem[
                tb_mx["Tb"].iloc[index_rem] < tb_rat["Tb"].iloc[index_rem] * 0.075
            ]
        ] = 0

        df = pd.DataFrame({"index": index}, index=ind)
        df = df.bfill(limit=120)
        df = df.ffill(limit=120)
        index = np.array(df["index"])
        index[(lev1["elevation_angle"][:] < 89.0) & (index != 0)] = 2

    return index, status
