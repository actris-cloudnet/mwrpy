"""Module for plotting."""

import glob
import locale
from datetime import date, datetime

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.pyplot import Figure
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    MultipleLocator,
    NullLocator,
)
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma, ndarray

from mwrpy.atmos import abs_hum, dir_avg, t_dew_rh
from mwrpy.plots.plot_meta import _COLORS, ATTRIBUTES
from mwrpy.plots.plot_utils import (
    _calculate_rolling_mean,
    _gap_array,
    _get_bit_flag,
    _get_freq_flag,
    _get_ret_flag,
    _get_unmasked_values,
    _nan_time_gaps,
    _read_location,
)
from mwrpy.utils import (
    isbit,
    read_config,
    read_nc_field_name,
    read_nc_fields,
    seconds2hours,
)


class Dimensions:
    """Dimensions of a generated figure in pixels."""

    width: int
    height: int
    margin_top: int
    margin_right: int
    margin_bottom: int
    margin_left: int

    def __init__(self, fig, axes, pad_inches: float | None = None):
        if pad_inches is None:
            pad_inches = rcParams["savefig.pad_inches"]

        tightbbox = (
            fig.get_tightbbox(fig.canvas.get_renderer())
            .padded(pad_inches)
            .transformed(Affine2D().scale(fig.dpi))
        )
        self.width = int(tightbbox.width)
        self.height = int(tightbbox.height)

        x0, y0, x1, y1 = (
            Bbox.union([ax.get_window_extent() for ax in axes])
            .translated(-tightbbox.x0, -tightbbox.y0)
            .extents
        )
        self.margin_top = int(self.height - round(y1))
        self.margin_right = int(self.width - round(x1) - 1)
        self.margin_bottom = int(round(y0) - 1)
        self.margin_left = int(round(x0))


def generate_figure(
    nc_file: str,
    field_names: list,
    show: bool = False,
    save_path: str | None = None,
    max_y: int = 5,
    ele_range: tuple[float, float] = (
        -1.0,
        91.0,
    ),
    pointing: int = 0,
    dpi: int = 120,
    image_name: str | None = None,
    sub_title: bool = True,
    title: bool = True,
) -> tuple[Dimensions, str] | None:
    """Generates a mwrpy figure.

    Args:
        nc_file (str): Input file.
        field_names (list): Variable names to be plotted.
        show (bool, optional): If True, shows the figure. Default is True.
        save_path (str, optional): Setting this path will save the figure (in the
            given path). Default is None, when the figure is not saved.
        max_y (int, optional): Upper limit in the plots (km). Default is 12.
        ele_range (tuple, optional): Range of elevation angles to be plotted.
        pointing (int, optional): Type of observation (0: single pointing, 1: BL scan)
        dpi (int, optional): Figure quality (if saved). Higher value means
            more pixels, i.e., better image quality. Default is 120.
        image_name (str, optional): Name (and full path) of the output image.
            Overrides the *save_path* option. Default is None.
        sub_title (bool, optional): Add subtitle to image. Default is True.
        title (bool, optional): Add title to image. Default is True.

    Returns:
        Dimensions of the generated figure in pixels.
        File name of the generated figure.

    Examples:
        >>> from mwrpy.plots import generate_figure
        >>> generate_figure('lev2_file.nc', ['lwp'])
    """
    valid_fields, valid_names = _find_valid_fields(nc_file, field_names)
    if len(valid_fields) > 0:
        fig, axes = _initialize_figure(len(valid_fields), dpi)

        for ax, field, name in zip(axes, valid_fields, valid_names):
            time = _read_time_vector(nc_file)
            elevation_range = ATTRIBUTES[name].ele
            if elevation_range is None:
                elevation_range = ele_range
            ax.set_facecolor(_COLORS["lightgray"])
            if name not in ("elevation_angle", "azimuth_angle"):
                time = _elevation_filter(nc_file, time, elevation_range)
                field = _elevation_filter(nc_file, field, elevation_range)
            if title:
                _set_title(ax, name, nc_file, "")
            is_height = _is_height_dimension(nc_file, name)
            if not is_height:
                source = ATTRIBUTES[name].source
                _plot_instrument_data(
                    ax,
                    field,
                    name,
                    source,
                    time,
                    fig,
                    nc_file,
                    elevation_range,
                    pointing,
                )
            else:
                ax_value = _read_ax_values(nc_file)
                ax_value = (time, ax_value[1])
                field, ax_value = _screen_high_altitudes(field, ax_value, max_y)
                _set_ax(ax, max_y)

                plot_type = ATTRIBUTES[name].plot_type
                if plot_type == "mesh":
                    _plot_colormesh_data(ax, field, name, ax_value, nc_file)

        case_date = _set_labels(fig, axes[-1], nc_file, sub_title)
        file_name = handle_saving(
            nc_file, image_name, save_path, show, case_date, valid_names
        )
        return Dimensions(fig, axes), file_name
    return None


def _mark_gaps(
    time: ndarray,
    data: ma.MaskedArray,
    max_allowed_gap: float = 1,
    mask_edge: int = 0,
) -> tuple:
    """Mark gaps in time and data."""
    assert time[0] >= 0
    assert time[-1] <= 24
    max_gap = max_allowed_gap / 60
    if not ma.is_masked(data):
        mask_new = np.zeros(data.shape)
    elif ma.all(data.mask) is ma.masked:
        mask_new = np.ones(data.shape)
    else:
        mask_new = np.copy(data.mask)
    data_new = ma.copy(data)
    time_new = np.copy(time)
    gap_indices = np.where(np.diff(time) > max_gap)[0]
    # for ia in range(mask_edge):
    #     gap_indices = np.unique(
    #         np.sort(np.append(gap_indices, [gap_indices - ia, gap_indices + ia]))
    #     )
    if data.ndim == 2:
        temp_array = np.zeros((2, data.shape[1]))
        temp_mask = np.ones((2, data.shape[1]))
    else:
        temp_array = np.zeros(2)
        temp_mask = np.ones(2)
    time_delta = 0.0
    for ind in np.sort(gap_indices)[::-1]:
        ind += 1
        data_new = np.insert(data_new, ind, temp_array, axis=0)
        mask_new = np.insert(mask_new, ind, temp_mask, axis=0)
        time_new = np.insert(time_new, ind, time[ind] - time_delta)
        time_new = np.insert(time_new, ind, time[ind - 1] + time_delta)
    if (time[0] - 0) > max_gap:
        data_new = np.insert(data_new, 0, temp_array, axis=0)
        mask_new = np.insert(mask_new, 0, temp_mask, axis=0)
        time_new = np.insert(time_new, 0, time[0] - time_delta)
        time_new = np.insert(time_new, 0, time_delta)
    if (24 - time[-1]) > max_gap:
        ind = mask_new.shape[0]
        data_new = np.insert(data_new, ind, temp_array, axis=0)
        mask_new = np.insert(mask_new, ind, temp_mask, axis=0)
        time_new = np.insert(time_new, ind, 24 - time_delta)
        time_new = np.insert(time_new, ind, time[-1] + time_delta)
    data_new.mask = mask_new
    return time_new, data_new


def handle_saving(
    nc_file: str,
    image_name: str | None,
    save_path: str | None,
    show: bool,
    case_date: date,
    field_names: list,
    fix: str = "",
) -> str:
    """Returns file name of plot."""
    site_name = _read_location(nc_file)
    if image_name:
        date_string = case_date.strftime("%Y%m%d")
        file_name = f"{save_path}{date_string}_{site_name}_{image_name}.png"
        plt.savefig(
            f"{save_path}{date_string}_{site_name}_{image_name}.png",
            bbox_inches="tight",
        )
    elif save_path:
        file_name = _create_save_name(save_path, case_date, field_names, fix)
        plt.savefig(file_name, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return file_name


def _set_labels(fig, ax, nc_file: str, sub_title: bool = True) -> date:
    """Sets labels and returns date of netCDF file."""
    ax.set_xlabel("Time (UTC)", fontsize=13)
    case_date = _read_date(nc_file)
    site_name = _read_location(nc_file)
    if sub_title:
        _add_subtitle(fig, case_date, site_name)
    return case_date


def _set_title(ax, field_name: str, nc_file, identifier: str = " from MWRpy"):
    """Sets title of plot."""
    if ATTRIBUTES[field_name].name:
        ax.set_title(f"{ATTRIBUTES[field_name].name}{identifier}", fontsize=14)
    else:
        ax.set_title(
            f"{read_nc_field_name(nc_file, field_name)}{identifier}", fontsize=14
        )


def _find_valid_fields(nc_file: str, names: list) -> tuple[list, list]:
    """Returns valid field names and corresponding data."""
    valid_names, valid_data = names[:], []
    with netCDF4.Dataset(nc_file) as nc:
        for name in names:
            if name in nc.variables:
                if not nc.variables[name][:].mask.all():
                    valid_data.append(nc.variables[name][:])
                else:
                    valid_names.remove(name)
            else:
                valid_names.remove(name)
    # if not valid_names:
    #     raise ValueError("No fields to be plotted")
    return valid_data, valid_names


def _is_height_dimension(full_path: str, var_name: str) -> bool:
    """Checks for height dimension in netCDF file."""
    with netCDF4.Dataset(full_path) as nc:
        is_height = "height" in nc.variables[var_name].dimensions
    return is_height


def _elevation_filter(full_path: str, data_field: ndarray, ele_range: tuple) -> ndarray:
    """Filters data for specified range of elevation angles."""
    with netCDF4.Dataset(full_path) as nc:
        if "elevation_angle" in nc.variables:
            elevation = read_nc_fields(full_path, "elevation_angle")
            if data_field.ndim > 1:
                data_field = data_field[
                    (elevation >= ele_range[0]) & (elevation <= ele_range[1]), :
                ]
            else:
                data_field = data_field[
                    (elevation >= ele_range[0]) & (elevation <= ele_range[1])
                ]
    return data_field


def _pointing_filter(
    full_path: str, data_field: ndarray, ele_range: tuple, status: int
) -> ndarray:
    """Filters data according to pointing flag."""
    with netCDF4.Dataset(full_path) as nc:
        if "pointing_flag" in nc.variables:
            pointing = read_nc_fields(full_path, "pointing_flag")
            pointing = _elevation_filter(full_path, pointing, ele_range)
            if data_field.ndim > 1:
                data_field = data_field[pointing == status, :]
            else:
                data_field = data_field[pointing == status]
    return data_field


def _initialize_figure(n_subplots: int, dpi) -> tuple[Figure, list[Axes]]:
    """Creates an empty figure according to the number of subplots."""
    fig, axes = plt.subplots(
        n_subplots, figsize=(16, 4 + (n_subplots - 1) * 4.8), dpi=dpi, facecolor="white"
    )
    fig.subplots_adjust(left=0.06, right=0.73)
    axes_list = [axes] if isinstance(axes, Axes) else axes.tolist()
    return fig, axes_list


def _read_ax_values(full_path: str) -> tuple[ndarray, ndarray]:
    """Returns time and height arrays."""
    time = read_nc_fields(full_path, "time")
    height = read_nc_fields(full_path, "height")
    height_km = height / 1000
    return time, height_km


def _read_time_vector(nc_file: str) -> ndarray:
    """Converts time vector to fraction hour."""
    with netCDF4.Dataset(nc_file) as nc:
        time = nc.variables["time"][:]
    return seconds2hours(time)


def _screen_high_altitudes(data_field: ndarray, ax_values: tuple, max_y: int) -> tuple:
    """Removes altitudes from 2D data that are not visible in the figure.
    Bug in pcolorfast causing effect to axis not noticing limitation while
    saving fig. This fixes that bug till pcolorfast does fixing themselves.

    Args:
        data_field (ndarray): 2D data array.
        ax_values (tuple): Time and height 1D arrays.
        max_y (int): Upper limit in the plots (km).
    """
    alt = ax_values[-1]
    if data_field.ndim > 1:
        ind = int((np.argmax(alt > max_y) or len(alt)) + 1)
        data_field = data_field[:, :ind]
        alt = alt[:ind]
    return data_field, (ax_values[0], alt)


def _set_ax(ax, max_y: float, ylabel: str | None = None, min_y: float = 0.0):
    """Sets ticks and tick labels for plt.imshow()."""
    ticks_x_labels = _get_standard_time_ticks()
    ax.set_ylim(min_y, max_y)
    ax.set_xticks(np.arange(0, 25, 4, dtype=int))
    ax.set_xticklabels(ticks_x_labels, fontsize=12)
    ax.set_ylabel("Height a.s.l. (km)", fontsize=13)
    ax.set_xlim(0, 24)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13)


def _get_standard_time_ticks(resolution: int = 4) -> list:
    """Returns typical ticks / labels for a time vector between 0-24h."""
    return [
        f"{int(i):02d}:00" if 24 > i > 0 else ""
        for i in np.arange(0, 24.01, resolution)
    ]


def _init_colorbar(plot, axis):
    """Returns colorbar."""
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def _read_date(nc_file: str) -> date:
    """Returns measurement date."""
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
    with netCDF4.Dataset(nc_file) as nc:
        case_date = datetime.strptime(nc.date, "%Y-%m-%d")
    return case_date


def _add_subtitle(fig, case_date: date, site_name: str):
    """Adds subtitle into figure."""
    text = _get_subtitle_text(case_date, site_name)
    fig.suptitle(
        text,
        fontsize=13,
        y=0.885,
        x=0.07,
        horizontalalignment="left",
        verticalalignment="bottom",
    )


def _get_subtitle_text(case_date: date, site_name: str) -> str:
    """Returns string with site name and date."""
    site_name = site_name.replace("-", " ")
    return f"{site_name}, {case_date.strftime('%-d %b %Y')}"


def _create_save_name(
    save_path: str, case_date: date, field_names: list, fix: str = ""
) -> str:
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{'_'.join(field_names)}{fix}.png"


def _plot_segment_data(ax, data: ma.MaskedArray, name: str, axes: tuple, nc_file: str):
    """Plots categorical 2D variable.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.
        nc_file (str): Input file.
    """
    if name == "tb_missing":
        cmap = ListedColormap(["#FFFFFF00", _COLORS["gray"]])
        ax.pcolor(*axes, data.T, cmap=cmap, shading="nearest", vmin=-0.5, vmax=1.5)
    elif name == "tb_qf":
        cmap = ListedColormap([_COLORS["lightgray"], _COLORS["darkgray"]])
        ax.pcolor(*axes, data.T, cmap=cmap, shading="nearest", vmin=-0.5, vmax=1.5)
    else:
        variables = ATTRIBUTES[name]
        assert variables.clabel is not None
        clabel = [x[0] for x in variables.clabel]
        cbar = [x[1] for x in variables.clabel]
        cmap = ListedColormap(cbar)
        x, y = axes[0], axes[1]
        x[1:] = x[1:] - np.diff(x)
        pl = ax.pcolor(
            x, y, data.T, cmap=cmap, shading="nearest", vmin=-0.5, vmax=len(cbar) - 0.5
        )
        ax.grid(axis="y")
        colorbar = _init_colorbar(pl, ax)
        colorbar.set_ticks(np.arange(len(clabel)))
        if name == "quality_flag_3":
            site = _read_location(nc_file)
            params = read_config(site, "params")
            clabel[2] = clabel[2] + " (" + str(params["TB_threshold"][1]) + " K)"
            clabel[1] = clabel[1] + " (" + str(params["TB_threshold"][0]) + " K)"
        colorbar.ax.set_yticklabels(clabel, fontsize=13)


def _plot_colormesh_data(ax, data_in: np.ndarray, name: str, axes: tuple, nc_file: str):
    """Plots continuous 2D variable.
    Creates only one plot, so can be used both one plot and subplot type of figs.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data_in (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.
        nc_file (str): Input file.
    """
    data = data_in.copy()
    variables = ATTRIBUTES[name]
    nbin = 7
    nlev1 = 31
    avg_time = 15 / 60
    if ATTRIBUTES[name].nlev:
        nlev = ATTRIBUTES[name].nlev
    else:
        nlev = 16

    assert variables.plot_range is not None

    if any(map(nc_file.__contains__, ["2P04", "2P07", "2P08"])):
        file_name = glob.glob(nc_file.rsplit("/", 1)[0] + "/MWR_2P02*")
        if len(file_name) == 0:
            file_name = glob.glob(nc_file.rsplit("/", 1)[0] + "/MWR_2P01*")
        tem_file = file_name[0]
    else:
        tem_file = nc_file

    if name == "potential_temperature":
        hum_file = nc_file.replace("2P07", "2P03")

    if name == "equivalent_potential_temperature":
        nbin = 9
        nlev1 = 41
        hum_file = nc_file.replace("2P08", "2P03")

    if name == "absolute_humidity":
        nbin = 6

    if name == "relative_humidity":
        assert isinstance(data_in, ma.MaskedArray)
        data[data_in.mask] = np.nan
        data[data > 1.0] = 1.001
        data[data < 0.0] = 0.0
        data_in[data_in > 1.0] = 1.001
        data_in[data_in < 0.0] = 0.0
        data *= 100.0
        data_in *= 100.0
        nbin = 6
        hum_file = nc_file.replace("2P04", "2P03")

    if "multi" in nc_file:
        hum_file = nc_file.replace("multi", "single")

    if name in (
        "relative_humidity",
        "potential_temperature",
        "equivalent_potential_temperature",
    ):
        hum_time = seconds2hours(read_nc_fields(hum_file, "time"))
        hum_flag = _get_ret_flag(hum_file, hum_time, "absolute_humidity")
        hum_flag = _calculate_rolling_mean(hum_time, hum_flag, win=avg_time)
        hum_flag = np.interp(axes[0], hum_time, hum_flag)
    else:
        hum_flag = np.zeros(len(axes[0]), np.int32)

    if variables.plot_type == "bit":
        cmap: Colormap = ListedColormap(str(variables.cbar))
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])
    else:
        cmap = plt.get_cmap(str(variables.cbar), nlev1)

    vmin, vmax = variables.plot_range

    if variables.cbar_ext in ("neither", "max"):
        data[data < vmin] = vmin

    if np.ma.median(np.diff(axes[0][:])) < avg_time:
        data = _calculate_rolling_mean(axes[0], data, win=avg_time)
        time, data = _mark_gaps(axes[0][:], ma.MaskedArray(data), 35, 10)
    else:
        time, data = _mark_gaps(
            axes[0][:],
            ma.MaskedArray(data_in),
            np.ma.median(np.diff(axes[0][:])) * 60.0 * 2,
            0,
        )

    ax.contourf(
        time,
        axes[1],
        data.T,
        levels=np.linspace(vmin, vmax, nlev1),
        cmap=cmap,
        extend=variables.cbar_ext,
        alpha=0.5,
    )

    if name in (
        "relative_humidity",
        "potential_temperature",
        "equivalent_potential_temperature",
    ):
        flag = _get_ret_flag(tem_file, axes[0], "temperature")
    else:
        flag = _get_ret_flag(tem_file, axes[0], name)
    if np.ma.median(np.diff(axes[0][:])) < avg_time:
        flag = _calculate_rolling_mean(axes[0], flag, win=avg_time)
        data_in[(flag > 0) | (hum_flag > 0), :] = np.nan
        data = _calculate_rolling_mean(axes[0], data_in, win=avg_time)
        time, data = _mark_gaps(axes[0][:], ma.MaskedArray(data), 35, 10)
    else:
        data_in[(flag > 0) | (hum_flag > 0), :] = np.nan
        val, cnt = np.unique(np.round(np.diff(axes[0][:]), 2), return_counts=True)
        time, data = _mark_gaps(
            axes[0][:],
            ma.MaskedArray(data_in),
            np.max(val[cnt > 10]) * 60.0 * 2,
            0,
        )

    if variables.cbar_ext in ("neither", "max"):
        data[data < vmin] = vmin

    pl = ax.contourf(
        time,
        axes[1],
        data.T,
        levels=np.linspace(vmin, vmax, nlev1),
        cmap=cmap,
        extend=variables.cbar_ext,
    )
    ds = int(np.round(len(time) * 0.05))
    assert isinstance(nlev, int)
    cp = ax.contour(
        time[ds : len(time) - ds],
        axes[1],
        data[ds : len(time) - ds, :].T,
        levels=np.linspace(vmin, vmax, nlev),
        colors="black",
        linewidths=0.0001,
    )
    cbl = plt.clabel(cp, fontsize=8)
    ta = np.array([])
    for lab in cbl:
        lab.set_verticalalignment("bottom")
        lab.set_fontweight("bold")
        if float(lab.get_text()) in ta:
            lab.set_visible(False)
        ta = np.append(ta, [float(lab.get_text())])
    ax.contour(
        time,
        axes[1],
        data.T,
        levels=np.linspace(vmin, vmax, nlev),
        colors="black",
        linewidths=0.8,
    )

    if variables.plot_type != "bit":
        colorbar = _init_colorbar(pl, ax)
        locator = colorbar.ax.yaxis.get_major_locator()
        # locator, formatter = colorbar._get_ticker_locator_formatter()
        locator.set_params(nbins=nbin)
        colorbar.update_ticks()
        colorbar.set_label(variables.clabel, fontsize=13)


def _plot_instrument_data(
    ax,
    data: ma.MaskedArray,
    name: str,
    product: str | None,
    time: ndarray,
    fig,
    nc_file: str,
    ele_range: tuple,
    pointing: int,
):
    """Calls plotting function for specified product."""
    if product == "int":
        _plot_int(ax, data, name, time, nc_file)
    elif product == "sta":
        _plot_sta(ax, data, name, time, nc_file)
    elif product in ("met", "met2"):
        _plot_met(ax, data, name, time, nc_file)
    elif product == "tb":
        _plot_tb(data, time, fig, nc_file, ele_range, pointing, name)
    elif product == "irt":
        _plot_irt(ax, data, name, time, nc_file)
    elif product == "qf":
        _plot_qf(data, time, fig, nc_file)
    elif product == "mqf":
        _plot_mqf(ax, data, time, nc_file)
    elif product == "sen":
        _plot_sen(ax, data, name, time, nc_file)
    elif product == "hkd":
        _plot_hkd(ax, data, name, time)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])


def _plot_hkd(ax, data_in: ndarray, name: str, time: ndarray):
    """Plot for housekeeping data."""
    time = _nan_time_gaps(time)
    if name == "t_amb":
        data_in[data_in == -999.0] = np.nan
        if (data_in[:, 0].all() is ma.masked) | (data_in[:, 1].all() is ma.masked):
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.plot(
                time,
                np.abs(data_in[:, 0] - data_in[:, 1]),
                color=_COLORS["darkgray"],
                label="Difference",
                linewidth=0.8,
            )
            _set_ax(
                ax,
                np.nanmax(np.abs(data_in[:, 0] - data_in[:, 1])) + 0.025,
                "Sensor absolute difference (K)",
                0.0,
            )
            ax.yaxis.set_label_position("right")
            ax.legend(loc="upper right")

        ax2 = ax.twinx()
        vmin, vmax = np.nanmin(data_in) - 1.0, np.nanmax(data_in) + 1.0
        ax2.plot(time, np.mean(data_in, axis=1), color="darkblue", label="Mean")
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        _set_ax(ax2, vmax, "Sensor mean (K)", vmin)
        if np.nanmax(np.abs(data_in[:, 0] - data_in[:, 1])) > 0.3:
            ax.plot(
                time,
                np.ones(len(time), np.float32) * 0.3,
                color="black",
                linewidth=0.8,
                label="Threshold (Diff.)",
            )
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        leg = ax2.legend(lines + lines2, labels + labels2, loc="upper right")
        ax.yaxis.tick_right()

    if name == "t_rec":
        ax.plot(time, data_in[:, 0], color="sienna", linewidth=0.8, label="Receiver 1")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax2 = ax.twinx()
        ax2.plot(
            time,
            data_in[:, 1],
            color=_COLORS["shockred"],
            linewidth=0.8,
            label="Receiver 2",
        )
        vmin1, vmax1 = np.nanmin(data_in[:, 0]) - 0.01, np.nanmax(data_in[:, 0]) + 0.01
        vmin2, vmax2 = np.nanmin(data_in[:, 1]) - 0.01, np.nanmax(data_in[:, 1]) + 0.01
        if vmax1 - vmin1 > vmax2 - vmin2:
            _set_ax(ax, vmax1, "Receiver 1 (K)", vmin1)
            _set_ax(
                ax2,
                vmax2 + ((vmax1 - vmin1) - (vmax2 - vmin2)) / 2,
                "Receiver 2 (K)",
                vmin2 - ((vmax1 - vmin1) - (vmax2 - vmin2)) / 2,
            )
        else:
            _set_ax(
                ax,
                vmax1 + ((vmax2 - vmin2) - (vmax1 - vmin1)) / 2,
                "Receiver 1 (K)",
                vmin1 - ((vmax2 - vmin2) - (vmax1 - vmin1)) / 2,
            )
            _set_ax(ax2, vmax2, "Receiver 2 (K)", vmin2)
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        leg = ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    if name == "t_sta":
        vmin, vmax = (
            0.0,
            np.nanmax([np.nanmax(data_in[:, 0]), np.nanmax(data_in[:, 1])])
            + 0.1 * np.nanmax([np.nanmax(data_in[:, 0]), np.nanmax(data_in[:, 1])]),
        )
        ax.plot(time, data_in[:, 0], color="sienna", linewidth=0.8, label="Receiver 1")
        ax.plot(
            time,
            data_in[:, 1],
            color=_COLORS["shockred"],
            linewidth=0.8,
            label="Receiver 2",
        )
        if vmax - 0.1 * vmax > 0.05:
            ax.plot(
                time,
                np.ones(len(time), np.float32) * 0.05,
                color="black",
                linewidth=0.8,
                label="Threshold",
            )
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        _set_ax(ax, vmax, "Mean absolute difference (K)", vmin)
        leg = ax.legend(loc="upper right")

    for legobj in leg.legend_handles:
        legobj.set_linewidth(2.0)


def _plot_sen(ax, data_in: ndarray, name: str, time: ndarray, nc_file: str):
    """Plot for azimuth and elevation angles."""
    variables = ATTRIBUTES[name]
    pointing_flag = read_nc_fields(nc_file, "pointing_flag")
    quality_flag = read_nc_fields(nc_file, "quality_flag")
    qf = _get_freq_flag(quality_flag, np.array([6]))
    assert variables.plot_range is not None
    vmin, vmax = variables.plot_range
    time1 = time[(pointing_flag == 1)]
    time1 = _nan_time_gaps(time1, 15.0 / 60.0)
    ax.plot(
        time[pointing_flag == 0],
        data_in[pointing_flag == 0],
        "--.",
        color=_COLORS["darkgreen"],
        label="single pointing",
        linewidth=0.8,
    )
    if name == "elevation_angle":
        time1 = time[(pointing_flag == 1) & (data_in > 0.0)]
        val, cnt = np.unique(np.round(np.diff(time1), 2), return_counts=True)
        if len(val) > 0:
            time1 = _nan_time_gaps(time1, np.max(val[cnt > 10]) * 2)
        else:
            time1 = _nan_time_gaps(time1, np.median(np.diff(time1)) * 2)
        ax.plot(
            time1,
            data_in[(pointing_flag == 1) & (data_in > 0.0)],
            "--.",
            alpha=0.75,
            color=_COLORS["green"],
            label="multiple pointing",
            linewidth=0.8,
        )
        ax.set_yticks(np.linspace(0, 90, 7))
    else:
        ax.plot(
            time1,
            data_in[(pointing_flag == 1)],
            "--.",
            alpha=0.75,
            color=_COLORS["green"],
            label="multiple pointing",
            linewidth=0.8,
        )
        ax.set_yticks(np.linspace(0, 360, 9))
    ax.plot(
        time[np.any(qf == 1, axis=1)],
        data_in[np.any(qf == 1, axis=1)],
        "r.",
        linewidth=1,
        label="sun_moon_in_beam",
    )
    ax.set_ylim((vmin, vmax))
    ax.legend(loc="upper right")
    _set_ax(ax, vmax, variables.ylabel, vmin)


def _plot_irt(ax, data_in: ma.MaskedArray, name: str, time: ndarray, nc_file: str):
    """Plot for infrared temperatures."""
    variables = ATTRIBUTES[name]
    assert variables.plot_range is not None
    vmin, vmax = variables.plot_range
    ir_wavelength = read_nc_fields(nc_file, "ir_wavelength")
    if not data_in[:, 0].mask.all():
        ax.plot(
            time,
            data_in[:, 0],
            "o",
            markersize=0.75,
            fillstyle="full",
            color="sienna",
            label=str(np.round(ir_wavelength[0] / 1e-6, 1)) + " µm",
        )
    if data_in.shape[1] > 1:
        if not data_in[:, 1].mask.all():
            ax.plot(
                time,
                data_in[:, 1],
                "o",
                markersize=0.75,
                fillstyle="full",
                color=_COLORS["shockred"],
                label=str(np.round(ir_wavelength[1] / 1e-6, 1)) + " µm",
            )
    ax.set_ylim((vmin, vmax))
    ax.legend(loc="upper right", markerscale=6)
    _set_ax(ax, vmax, variables.ylabel, vmin)


def _plot_mqf(ax, data_in: ma.MaskedArray, time: ndarray, nc_file: str):
    """Plot for quality flags of meteorological sensors."""
    qf = _get_bit_flag(data_in, np.arange(6))
    _plot_segment_data(
        ax,
        ma.MaskedArray(qf),
        "met_quality_flag",
        (time, np.linspace(0.5, 5.5, 6)),
        nc_file,
    )
    ax.set_yticks(np.arange(6))
    ax.yaxis.set_ticklabels([])
    _set_ax(ax, 6, "")
    ax.set_title(ATTRIBUTES["met_quality_flag"].name)


def _plot_qf(data_in: ndarray, time: ndarray, fig, nc_file: str):
    """Plot for Level 1 quality flags."""
    site = _read_location(nc_file)
    params = read_config(site, "params")

    fig.clear()
    nsub = 4
    h_ratio = [0.1, 0.3, 0.3, 0.3]
    if params["flag_status"][3] == 1:
        nsub = 3
        h_ratio = [0.2, 0.4, 0.4]
    fig, axs = plt.subplots(
        nsub, 1, figsize=(12.52, 16), dpi=120, facecolor="w", height_ratios=h_ratio
    )
    assert not isinstance(axs, Axes)
    frequency = read_nc_fields(nc_file, "frequency")

    qf = _get_bit_flag(data_in[:, 0], np.array([5, 6]))
    _plot_segment_data(
        axs[0],
        ma.MaskedArray(qf),
        "quality_flag_0",
        (time, np.linspace(0.5, 1.5, 2)),
        nc_file,
    )
    axs[0].set_yticks(np.arange(2))
    axs[0].yaxis.set_ticklabels([])
    axs[0].set_facecolor(_COLORS["lightgray"])
    _set_ax(axs[0], 2, "")
    axs[0].set_title(ATTRIBUTES["quality_flag_0"].name)

    case_date = _read_date(nc_file)
    gtim = _gap_array(time, case_date)

    qf1 = _get_freq_flag(data_in[:, np.array(params["receiver"]) == 1], np.array([4]))
    qf2 = _get_freq_flag(data_in[:, np.array(params["receiver"]) == 2], np.array([4]))
    qf = np.column_stack((qf1 - 1, qf2 + 1))
    _plot_segment_data(
        axs[1],
        ma.MaskedArray(qf),
        "quality_flag_1",
        (time, np.linspace(0.5, len(frequency) - 0.5, len(frequency))),
        nc_file,
    )
    axs[1].set_title(ATTRIBUTES["quality_flag_1"].name)

    if params["flag_status"][3] == 0:
        qf = _get_freq_flag(data_in, np.array([3]))
        if len(gtim) > 0:
            time_i, data_g = (
                np.linspace(time[0], time[-1], len(time)),
                np.zeros((len(time), 20), np.float32),
            )
            for ig, _ in enumerate(gtim[:, 0]):
                xind = np.where((time_i >= gtim[ig, 0]) & (time_i <= gtim[ig, 1]))
                data_g[xind, :] = 1.0
            _plot_segment_data(
                axs[2],
                ma.MaskedArray(data_g),
                "tb_qf",
                (time_i, np.linspace(0.5, 20.0 - 0.5, 20)),
                nc_file,
            )
        _plot_segment_data(
            axs[2],
            ma.MaskedArray(qf),
            "quality_flag_2",
            (time, np.linspace(0.5, len(frequency) - 0.5, len(frequency))),
            nc_file,
        )
        axs[2].set_title(ATTRIBUTES["quality_flag_2"].name)

    qf = _get_freq_flag(data_in, np.array([1, 2]))
    if len(gtim) > 0:
        time_i, data_g = (
            np.linspace(time[0], time[-1], len(time)),
            np.zeros((len(time), 20), np.float32),
        )
        for ig, _ in enumerate(gtim[:, 0]):
            xind = np.where((time_i >= gtim[ig, 0]) & (time_i <= gtim[ig, 1]))
            data_g[xind, :] = 1.0
        _plot_segment_data(
            axs[nsub - 1],
            ma.MaskedArray(data_g),
            "tb_qf",
            (time_i, np.linspace(0.5, 20.0 - 0.5, 20)),
            nc_file,
        )
    _plot_segment_data(
        axs[nsub - 1],
        ma.MaskedArray(qf),
        "quality_flag_3",
        (time, np.linspace(0.5, len(frequency) - 0.5, len(frequency))),
        nc_file,
    )
    axs[nsub - 1].set_title(ATTRIBUTES["quality_flag_3"].name)

    offset = ScaledTranslation(0 / 72, 8 / 72, fig.dpi_scale_trans)
    if params["flag_status"][3] == 1:
        offset = ScaledTranslation(0 / 72, 1 / 3, fig.dpi_scale_trans)
    for i in np.array(np.linspace(1, nsub - 1, nsub - 1), np.int32):
        axs[i].set_yticks(range(len(frequency)))
        axs[i].set_yticklabels(frequency)
        axs[i].set_facecolor(_COLORS["lightgray"])
        for label in axs[i].yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        _set_ax(axs[i], len(frequency), "Frequency [GHz]")
        ind1, ind2 = axs[i].get_xlim()
        axs[i].plot(
            np.linspace(ind1, ind2, len(time)),
            np.ones(len(time)) * np.sum(np.array(params["receiver"]) == 1),
            "k-",
            linewidth=1,
        )
    _set_labels(fig, axs[-1], nc_file)


def _plot_tb(
    data_in: ndarray,
    time: ndarray,
    fig,
    nc_file: str,
    ele_range: tuple,
    pointing: int,
    name: str,
):
    """Plot for microwave brightness temperatures."""
    site = _read_location(nc_file)
    params = read_config(site, "params")
    frequency = read_nc_fields(nc_file, "frequency")
    quality_flag = read_nc_fields(nc_file, "quality_flag")
    if name == "tb_spectrum":
        tb = read_nc_fields(nc_file, "tb")
        tb = _elevation_filter(nc_file, tb, ele_range)
        quality_flag[~isbit(quality_flag, 3)] = 0
        data_in = tb - data_in

    data_in = _pointing_filter(nc_file, data_in, ele_range, pointing)
    time = _pointing_filter(nc_file, time, ele_range, pointing)
    quality_flag = _elevation_filter(nc_file, quality_flag, ele_range)
    quality_flag = _pointing_filter(nc_file, quality_flag, ele_range, pointing)

    fig.clear()
    fig, axs = plt.subplots(
        7,
        len(frequency) % 6,
        figsize=(13, 16),
        facecolor="w",
        edgecolor="k",
        sharex="col",
        dpi=120,
    )
    assert not isinstance(axs, Axes)
    fig.subplots_adjust(hspace=0.035, wspace=0.15)
    if pointing == 0:
        ylabel = "Brightness Temperature (single pointing) [K]"
    else:
        ylabel = "Brightness Temperature (multiple pointing) [K]"
    if name == "tb":
        fig.text(
            0.06,
            0.5,
            ylabel,
            va="center",
            rotation="vertical",
            fontsize=20,
        )
        if len(frequency) % 6 == 2:
            fig.text(0.445, 0.09, "flagged data", va="center", fontsize=20, color="r")
        else:
            fig.text(0.795, 0.09, "flagged data", va="center", fontsize=20, color="r")
    else:
        fig.text(
            0.06,
            0.5,
            "Brightness Temperature (Observed - Retrieved) [K]",
            va="center",
            rotation="vertical",
            fontsize=20,
        )
        if len(frequency) % 6 == 2:
            fig.text(
                0.37,
                0.085,
                "spectral consistency failed",
                va="center",
                fontsize=20,
                color="r",
            )
        else:
            fig.text(
                0.635,
                0.085,
                "spectral consistency failed",
                va="center",
                fontsize=20,
                color="r",
            )

    if axs.ndim > 1:
        axs[0, 0].set_title(
            "Receiver 1 Channels", fontsize=15, color=_COLORS["darkgray"], loc="right"
        )
        axs[0, 1].set_title(
            "Receiver 2 Channels", fontsize=15, color=_COLORS["darkgray"], loc="right"
        )
    else:
        axs[np.where(np.array(params["receiver"]) == 1)[0][0]].set_title(
            "Receiver 1",
            fontsize=15,
            color=_COLORS["darkgray"],
            loc="right",
            rotation="vertical",
            x=1.03,
            y=0.05,
        )
        axs[np.where(np.array(params["receiver"]) == 2)[0][0]].set_title(
            "Receiver 2",
            fontsize=15,
            color=_COLORS["darkgray"],
            loc="right",
            rotation="vertical",
            x=1.03,
            y=0.05,
        )

    trans = ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    tb_m: np.ndarray = np.array([])
    tb_s: np.ndarray = np.array([])

    for i, axi in enumerate(axs.T.flatten()):
        no_flag = np.where(quality_flag[:, i] == 0)[0]
        if len(np.array(no_flag)) == 0:
            no_flag = np.arange(len(time))
        tb_m = np.append(tb_m, [np.nanmean(data_in[no_flag, i])])  # TB mean
        tb_s = np.append(tb_s, [np.nanstd(data_in[no_flag, i])])  # TB std
        axi.plot(
            time,
            np.ones(len(time)) * tb_m[i],
            "--",
            color=_COLORS["darkgray"],
            linewidth=1,
        )
        axi.plot(time, data_in[:, i], "ko", markersize=0.75, fillstyle="full")
        flag = np.where(quality_flag[:, i] > 0)[0]
        axi.plot(time[flag], data_in[flag, i], "ro", markersize=0.75, fillstyle="full")
        axi.set_facecolor(_COLORS["lightgray"])
        if len(data_in) > 0:
            dif = np.nanmax(data_in[no_flag, i]) - np.nanmin(data_in[no_flag, i])
            _set_ax(
                axi,
                np.nanmax(data_in[no_flag, i]) + 0.25 * dif,
                "",
                np.nanmin(data_in[no_flag, i]) - 0.25 * dif,
            )
        else:
            _set_ax(
                axi,
                100.0,
                "",
                0.0,
            )

        if i in (6, 13):
            _set_labels(fig, axi, nc_file)
        axi.text(
            0.05,
            0.9,
            str(frequency[i]) + " GHz",
            transform=axi.transAxes + trans,
            color=_COLORS["darkgray"],
            fontweight="bold",
        )
        axi.text(
            0.55,
            0.9,
            str(round(tb_m[i], 2)) + " +/- " + str(round(tb_s[i], 2)) + " K",
            transform=axi.transAxes + trans,
            color=_COLORS["darkgray"],
            fontweight="bold",
        )

    # TB mean
    axa = fig.add_subplot(121)
    axa.set_position([0.125, -0.05, 0.72, 0.125])
    axa.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    axa.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
    for pos in ["right", "top", "bottom", "left"]:
        axa.spines[pos].set_visible(False)

    if name == "tb":
        axa.set_facecolor(_COLORS["lightgray"])
        axaK = fig.add_subplot(121)
        axaK.set_position([0.125, -0.05, 0.36, 0.125])
        axaK.plot(frequency, tb_m, "ko", markerfacecolor="k", markersize=4)
        axaK.errorbar(
            frequency, tb_m, yerr=tb_s, xerr=None, linestyle="", capsize=8, color="k"
        )
        axaK.set_xticks(frequency)
        axaK.set_xticklabels(axaK.get_xticks(), rotation=30)
        axaK.set_xlim(
            [
                np.floor(np.nanmin(frequency[np.array(params["receiver"]) == 1]) - 0.1),
                np.ceil(np.nanmax(frequency[np.array(params["receiver"]) == 1]) + 0.1),
            ]
        )
        minv = np.nanmin(
            tb_m[np.array(params["receiver"]) == 1]
            - tb_s[np.array(params["receiver"]) == 1]
        )
        maxv = np.nanmax(
            tb_m[np.array(params["receiver"]) == 1]
            + tb_s[np.array(params["receiver"]) == 1]
        )
        axaK.set_ylim([np.nanmax([0, minv - 0.05 * minv]), maxv + 0.05 * maxv])
        axaK.tick_params(axis="both", labelsize=12)
        axaK.set_facecolor(_COLORS["lightgray"])
        axaK.plot(
            frequency[np.array(params["receiver"]) == 1],
            tb_m[np.array(params["receiver"]) == 1],
            "k-",
            linewidth=2,
        )

        axaV = fig.add_subplot(122)
        axaV.set_position([0.54, -0.05, 0.36, 0.125])
        axaV.plot(frequency, tb_m, "ko", markerfacecolor="k", markersize=4)
        axaV.errorbar(
            frequency, tb_m, yerr=tb_s, xerr=None, linestyle="", capsize=8, color="k"
        )
        axaV.set_xticks(frequency)
        axaV.set_xticklabels(axaV.get_xticks(), rotation=30)
        axaV.set_xlim(
            [
                np.floor(np.nanmin(frequency[np.array(params["receiver"]) == 2]) - 0.1),
                np.ceil(np.nanmax(frequency[np.array(params["receiver"]) == 2]) + 0.1),
            ]
        )
        minv = np.nanmin(
            tb_m[np.array(params["receiver"]) == 2]
            - tb_s[np.array(params["receiver"]) == 2]
        )
        maxv = np.nanmax(
            tb_m[np.array(params["receiver"]) == 2]
            + tb_s[np.array(params["receiver"]) == 2]
        )
        axaV.set_ylim([np.nanmax([0, minv - 0.05 * minv]), maxv + 0.05 * maxv])
        axaV.tick_params(axis="both", labelsize=12)
        axaV.set_facecolor(_COLORS["lightgray"])
        axaV.plot(
            frequency[np.array(params["receiver"]) == 2],
            tb_m[np.array(params["receiver"]) == 2],
            "k-",
            linewidth=2,
        )

        axaK.spines["right"].set_visible(False)
        axaV.spines["left"].set_visible(False)
        axaV.yaxis.tick_right()
        d = 0.015
        axaK.plot(
            (1 - d, 1 + d), (-d, +d), transform=axaK.transAxes, color="k", clip_on=False
        )
        axaK.plot(
            (1 - d, 1 + d),
            (1 - d, 1 + d),
            transform=axaK.transAxes,
            color="k",
            clip_on=False,
        )
        axaV.plot(
            (-d, +d), (1 - d, 1 + d), transform=axaV.transAxes, color="k", clip_on=False
        )
        axaV.plot(
            (-d, +d), (-d, +d), transform=axaV.transAxes, color="k", clip_on=False
        )
        axaK.set_ylabel("Brightness Temperature [K]", fontsize=12)
        axaV.text(
            -0.08,
            0.9,
            "TB daily means +/- standard deviation",
            fontsize=13,
            horizontalalignment="center",
            transform=axaV.transAxes,
            color=_COLORS["darkgray"],
            fontweight="bold",
        )
        axaV.text(
            -0.08,
            -0.35,
            "Frequency [GHz]",
            fontsize=13,
            horizontalalignment="center",
            transform=axaV.transAxes,
        )

    else:
        tb_m = np.ones((len(time), len(params["receiver_nb"]))) * np.nan
        axa = fig.subplots(1, 2)
        ticks_x_labels = _get_standard_time_ticks()
        axa[0].set_ylabel("Mean absolute difference [K]", fontsize=12)

        rain_flag = read_nc_fields(nc_file, "quality_flag")
        rain_flag = _elevation_filter(nc_file, rain_flag, ele_range)
        rain_flag = _pointing_filter(nc_file, rain_flag, ele_range, pointing)

        for irec, rec in enumerate(params["receiver_nb"]):
            axa[irec].set_position([0.125 + irec * 0.415, -0.05, 0.36, 0.125])
            no_flag = np.where(
                np.sum(quality_flag[:, np.array(params["receiver"]) == rec], axis=1)
                == 0
            )[0]
            if len(no_flag) == 0:
                no_flag = np.arange(len(time))
            tb_m[:, irec] = ma.mean(
                np.abs(data_in[:, np.array(params["receiver"]) == rec]), axis=1
            )
            axa[irec].plot(
                time,
                tb_m[:, irec],
                "o",
                color="black",
                markersize=0.75,
                fillstyle="full",
            )
            axa[irec].set_facecolor(_COLORS["lightgray"])
            flag = np.where(
                np.sum(quality_flag[:, np.array(params["receiver"]) == rec], axis=1) > 0
            )[0]
            axa[irec].plot(
                time[flag], tb_m[flag, irec], "ro", markersize=0.75, fillstyle="full"
            )
            axa[irec].set_xticks(np.arange(0, 25, 4, dtype=int))
            axa[irec].set_xticklabels(ticks_x_labels, fontsize=12)
            axa[irec].set_xlim(0, 24)
            axa[irec].set_xlabel("Time (UTC)", fontsize=12)
            axa[irec].set_ylim([0, np.nanmax(tb_m[no_flag, irec], initial=0.0) + 0.5])

            if len(np.where(isbit(rain_flag[:, 0], 5))[0]) > 0:
                data_g = np.zeros((len(time), 10), np.float32)
                data_g[isbit(rain_flag[:, 0], 5), :] = 1.0
                _plot_segment_data(
                    axa[irec],
                    ma.MaskedArray(data_g),
                    "tb_missing",
                    (
                        time,
                        np.linspace(
                            0, np.nanmax(tb_m[no_flag, irec], initial=0.0) + 0.5, 10
                        ),
                    ),
                    nc_file,
                )
                handles, labels = axa[irec].get_legend_handles_labels()
                handles.append(Patch(facecolor=_COLORS["gray"]))
                labels.append("rain_detected")
                axa[irec].legend(handles, labels, loc="upper right")


def _plot_met(ax, data_in: ndarray, name: str, time: ndarray, nc_file: str):
    """Plot for meteorological sensors."""
    ylabel = ATTRIBUTES[name].ylabel
    if name == "rainfall_rate":
        data_in *= 1000.0 * 3600.0
    if name == "air_pressure":
        data_in /= 100.0
    if name == "relative_humidity":
        data_in *= 100.0
        ylabel = "relative humidity (%)"
        ax.set_title("Relative and absolute humidity", fontsize=14)
    data, time = _get_unmasked_values(ma.MaskedArray(data_in), time)
    if name == "wind_direction":
        spd = read_nc_fields(nc_file, "wind_speed")
        rolling_mean, width = dir_avg(time, spd, data)
        rolling_mean = np.interp(
            time, time[int(width / 2 - 1) : int(-width / 2)], rolling_mean
        )
        ax.set_yticks(np.linspace(0, 360, 9))
    else:
        rolling_mean = _calculate_rolling_mean(time, data)

    time = _nan_time_gaps(time)

    if name not in ("rainfall_rate", "air_temperature", "relative_humidity"):
        ax.plot(time, data, ".", alpha=0.8, color=_COLORS["darksky"], markersize=1)
        ax.plot(
            time, rolling_mean, "o", fillstyle="full", color="darkblue", markersize=3
        )
    plot_range = ATTRIBUTES[name].plot_range
    assert plot_range is not None
    vmin, vmax = plot_range
    if name == "air_pressure":
        vmin, vmax = np.nanmin(data) - 1.0, np.nanmax(data) + 1.0
    if name in ("wind_speed", "rainfall_rate"):
        vmin, vmax = 0.0, np.nanmax([np.nanmax(data) + 1.0, 2.0])

    _set_ax(ax, vmax, ylabel, min_y=vmin)
    ax.grid(True)

    if name == "air_temperature":
        ax.plot(time, data, ".", alpha=0.8, color=_COLORS["darksky"], markersize=1)
        ax.plot(
            time,
            rolling_mean,
            "o",
            fillstyle="full",
            color="darkblue",
            markersize=3,
            label="Temperature",
        )
        rh = read_nc_fields(nc_file, "relative_humidity")
        rh = _elevation_filter(nc_file, rh, ele_range=(-1.0, 91.0))
        t_d = t_dew_rh(data, rh)
        rolling_mean = _calculate_rolling_mean(time, t_d)
        ax.plot(
            time,
            t_d,
            ".",
            alpha=0.8,
            color=_COLORS["blue"],
            markersize=1,
        )
        ax.plot(
            time,
            rolling_mean,
            "o",
            fillstyle="full",
            color=_COLORS["darkgray"],
            markersize=3,
            label="Dewpoint",
        )
        vmin, vmax = ma.min([data, t_d]) - 1.0, ma.max([data, t_d]) + 1.0
        ax.legend(loc="upper right", markerscale=2)
        _set_ax(ax, vmax, ylabel, min_y=vmin)
        ax.grid(True)

    if name == "relative_humidity":
        T = read_nc_fields(nc_file, "air_temperature")
        T = _elevation_filter(nc_file, T, ele_range=(-1.0, 91.0))
        q = abs_hum(T, data / 100.0)
        rolling_mean2 = _calculate_rolling_mean(time, q)
        ax2 = ax.twinx()
        ax2.plot(
            time,
            q,
            ".",
            alpha=0.8,
            color=_COLORS["blue"],
            markersize=1,
        )
        _set_ax(
            ax2,
            np.nanmax(
                [
                    np.nanmax(q) + 0.1 * np.nanmax(q),
                    0.003,
                ]
            ),
            "absolute humidity (kg m$^{-3}$)",
            min_y=np.nanmax(
                [
                    np.nanmin(q) - 0.1 * np.nanmin(q),
                    0.0,
                ]
            ),
        )
        ax2.plot(
            time,
            rolling_mean2,
            "o",
            fillstyle="full",
            color=_COLORS["darkgray"],
            markersize=3,
            label="Abs. hum.",
        )

        yl = ax.get_ylim()
        yl2 = ax2.get_ylim()

        ticks = _calculate_ticks(ax.get_yticks(), yl, yl2)

        ax2.yaxis.set_major_locator(FixedLocator(ticks))
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax3 = ax.twinx()
        ax3.plot(time, data, ".", alpha=0.8, color=_COLORS["darksky"], markersize=1)
        ax3.plot(
            time,
            rolling_mean,
            "o",
            fillstyle="full",
            color="darkblue",
            markersize=3,
            label="Rel. hum.",
        )
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(lines3 + lines2, labels3 + labels2, loc="upper right", markerscale=2)
        _set_ax(ax3, vmax, "", min_y=vmin)
        ax3.set_yticklabels([])
        ax3.set_frame_on(False)

    if name == "rainfall_rate":
        ax2 = ax.twinx()
        ax2.plot(
            time,
            np.cumsum(data) / 3600.0,
            color=_COLORS["darkgray"],
            linewidth=2.0,
        )
        _set_ax(
            ax2,
            np.nanmax(
                [
                    np.nanmax(np.cumsum(data) / 3600.0)
                    + 0.1 * np.nanmax(np.cumsum(data) / 3600.0),
                    0.004,
                ]
            ),
            "accum. amount (mm)",
            min_y=0.0,
        )
        if np.nanmax(data) <= 2.0:
            minorLocator = MultipleLocator(0.5)
            ax.yaxis.set_major_locator(minorLocator)
        yl = ax.get_ylim()
        yl2 = ax2.get_ylim()

        ticks = _calculate_ticks(ax.get_yticks(), yl, yl2)

        ax2.yaxis.set_major_locator(FixedLocator(ticks))
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        assert ylabel is not None
        _set_ax(ax, vmax, "rainfall rate (" + ylabel + ")", min_y=vmin)
        ax3 = ax.twinx()
        ax3.plot(time, data, ".", alpha=0.8, color=_COLORS["darksky"], markersize=1)
        ax3.plot(
            time, rolling_mean, "o", fillstyle="full", color="darkblue", markersize=3
        )
        _set_ax(ax3, vmax, "", min_y=vmin)
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax3.set_frame_on(False)


def _calculate_ticks(x, yl, yl2):
    return yl2[0] + (x - yl[0]) / (yl[1] - yl[0]) * (yl2[1] - yl2[0])


def _plot_int(ax, data_in: ma.MaskedArray, name: str, time: ndarray, nc_file: str):
    """Plot for integrated quantities (LWP, IWV)."""
    flag = _get_ret_flag(nc_file, time, name)
    data0, time0 = data_in[flag == 0], time[flag == 0]
    if len(data0) == 0:
        data0, time0 = data_in, time
    plot_range = ATTRIBUTES[name].plot_range
    assert plot_range is not None
    vmin, vmax = plot_range
    if name == "iwv":
        vmin, vmax = np.nanmin(data0) - 1.0, np.nanmax(data0) + 1.0
    else:
        vmax = np.min([np.nanmax(data0) + 0.05, vmax])
        vmin = np.max([np.nanmin(data0) - 0.05, vmin])
    _set_ax(ax, vmax, ATTRIBUTES[name].ylabel, min_y=vmin)

    flag_tmp = _calculate_rolling_mean(time, flag, win=1 / 60)
    data_f = np.zeros((len(flag_tmp), 10), np.float32)
    data_f[flag_tmp > 0, :] = 1.0
    cmap = ListedColormap([_COLORS["lightgray"], _COLORS["gray"]])
    norm = BoundaryNorm([0, 1, 2], cmap.N)
    ax.pcolormesh(
        time,
        np.linspace(vmin, vmax, 10),
        data_f.T,
        cmap=cmap,
        norm=norm,
    )

    case_date = _read_date(nc_file)
    gtim = _gap_array(time, case_date, 15.0 / 60.0)
    if len(gtim) > 0:
        time_i, data_g = (
            np.linspace(time[0], time[-1], len(time)),
            np.zeros((len(time), 10), np.float32),
        )
        for ig, _ in enumerate(gtim[:, 0]):
            xind = np.where((time_i >= gtim[ig, 0]) & (time_i <= gtim[ig, 1]))
            data_g[xind, :] = 1.0

        _plot_segment_data(
            ax,
            ma.MaskedArray(data_g),
            "tb_missing",
            (time_i, np.linspace(vmin, vmax, 10)),
            nc_file,
        )

    ax.plot(time, data_in, ".", color="royalblue", markersize=1)
    ax.axhline(linewidth=0.8, color="k")
    rolling_mean = _calculate_rolling_mean(time0, data0)
    time0 = _nan_time_gaps(time0)
    ax.plot(
        time0,
        rolling_mean,
        color="sienna",
        linewidth=2.0,
    )
    ax.plot(
        time0,
        rolling_mean,
        color="wheat",
        linewidth=0.6,
    )


def _plot_sta(ax, data_in: ma.MaskedArray, name: str, time: ndarray, nc_file: str):
    """Plot for stability indices."""
    flag = _get_ret_flag(nc_file, time, "stability")
    data0, time0 = data_in[flag == 0], time[flag == 0]
    if len(data0) == 0:
        data0, time0 = data_in, time
    plot_range = ATTRIBUTES[name].plot_range
    assert plot_range is not None

    rolling_mean = _calculate_rolling_mean(time0, data0, win=0.25)
    time0 = _nan_time_gaps(time0)
    vmin, vmax = plot_range
    vmax = np.min([np.nanmax(rolling_mean) + 0.05, vmax])
    vmin = np.max([np.nanmin(rolling_mean) - 0.05, vmin])
    _set_ax(ax, vmax, ATTRIBUTES[name].ylabel, min_y=vmin)
    _set_title(ax, name, nc_file, "")

    limit_cape = [1500, 300, 1, 0]
    limit_k_index = [35, 30, 1, 0]
    limit_total_totals = [53, 48, 1, 0]
    limit_lifted_index = [0, -3, 0, 1]
    limit_showalter_index = [2, -2, 0, 1]
    limit_ko_index = [6, 2, 0, 1]

    probability = ["Low", "High"]
    l_color = ["#3cb371", "#E64A23"]

    if vmin < eval("limit_" + name)[0] < vmax:
        ax.plot(
            time,
            np.ones(len(time)) * eval("limit_" + name)[0],
            color=l_color[eval("limit_" + name)[2]],
        )
        ax.text(
            0.5,
            eval("limit_" + name)[0] + 0.025 * (vmax - vmin),
            probability[eval("limit_" + name)[2]],
        )
    if vmin < eval("limit_" + name)[1] < vmax:
        ax.plot(
            time,
            np.ones(len(time)) * eval("limit_" + name)[1],
            color=l_color[eval("limit_" + name)[3]],
        )
        ax.text(
            0.5,
            eval("limit_" + name)[1] - 0.05 * (vmax - vmin),
            probability[eval("limit_" + name)[3]],
        )
    if vmax < np.min(eval("limit_" + name)[:2]):
        ax.text(0.5, vmax - 0.05 * (vmax - vmin), probability[eval("limit_" + name)[3]])
    if vmin > np.max(eval("limit_" + name)[:2]):
        ax.text(0.5, vmin + 0.05 * (vmax - vmin), probability[eval("limit_" + name)[2]])

    ax.plot(
        time0,
        rolling_mean,
        color="darkblue",
        linewidth=4.0,
    )
    ax.plot(
        time0,
        rolling_mean,
        color="lightblue",
        linewidth=1.2,
    )

    flag_tmp = _calculate_rolling_mean(time, flag, win=1 / 60)
    data_f = np.zeros((len(flag_tmp), 10), np.float32)
    data_f[flag_tmp > 0, :] = 1.0
    cmap = ListedColormap([_COLORS["lightgray"], _COLORS["gray"]])
    norm = BoundaryNorm([0, 1, 2], cmap.N)
    ax.pcolormesh(
        time,
        np.linspace(vmin, vmax, 10),
        data_f.T,
        cmap=cmap,
        norm=norm,
    )

    case_date = _read_date(nc_file)
    gtim = _gap_array(time, case_date, 15.0 / 60.0)
    if len(gtim) > 0:
        time_i, data_g = (
            np.linspace(time[0], time[-1], len(time)),
            np.zeros((len(time), 10), np.float32),
        )
        for ig, _ in enumerate(gtim[:, 0]):
            xind = np.where((time_i >= gtim[ig, 0]) & (time_i <= gtim[ig, 1]))
            data_g[xind, :] = 1.0

        _plot_segment_data(
            ax,
            ma.MaskedArray(data_g),
            "tb_missing",
            (time_i, np.linspace(vmin, vmax, 10)),
            nc_file,
        )
