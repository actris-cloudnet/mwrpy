"""Module for monthly statistic plots"""

import locale
import os
from calendar import month_abbr, monthrange
from datetime import date, timedelta
from itertools import groupby

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from numpy import ma

from mwrpy.plots.generate_plots import Dimensions, _initialize_figure
from mwrpy.plots.plot_utils import annotate_heatmap, heatmap
from mwrpy.plots.stat_meta import _COLORS, ATTRIBUTES
from mwrpy.utils import append_data, isbit, read_yaml_config


def generate_stat(
    site: str,
    names: list,
    year: str,
    image_name: str,
    save_path: str = None,
    show: bool = False,
    dpi: int = 120,
    title: bool = True,
) -> str:
    """Generates a mwrpy statistic figure.
    Args:
        site (str): Name of site.
        names (list): Variable names to be plotted.
        year (str): Year to be plotted.
        image_name (str): Name for image file.
        save_path (str, optional): Setting this path will save the figure (in the
            given path). Default is None, when the figure is not saved.
        show (bool, optional): If True, shows the figure. Default is True.
        dpi (int, optional): Figure quality (if saved). Higher value means
            more pixels, i.e., better image quality. Default is 120.
        title (bool, optional): Add title to image. Default is True.
    Returns:
        Dimensions of the generated figure in pixels.
        File name of the generated figure.
    Examples:
        >>> from plots import generate_stat
        >>> generate_stat([2020, 2021], 'data_availability')
    """

    fig, axes = _initialize_figure(len(names), dpi)
    global_attributes, params = read_yaml_config(site)
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
    for ax, name in zip(axes, names):
        stat, labels, bar_txt = calc_stat(fig, ax, year, name, params, global_attributes)
        if title:
            _set_title(ax, name)
        plot_type = ATTRIBUTES[name].plot_type
        if plot_type == "bar":
            _plot_bar(ax, name, year, stat, labels, bar_txt, params)
        if plot_type == "heat":
            _plot_heat_map(ax, name, year, stat, labels, bar_txt)
        if plot_type == "lines_rec":
            _plot_lines_rec(ax, name, stat, labels)
        if plot_type == "lines_amb":
            _plot_lines_amb(ax, name, stat, labels)
        _add_subtitle(fig, year, site)
    file_name = handle_saving(site, image_name, save_path, year, show)
    return file_name


def handle_saving(
    site_name: str,
    image_name: str | None,
    save_path: str | None,
    year: str | None,
    show: bool,
) -> str:
    """Returns file name of plot."""
    file_name = f"{save_path}{year}_{site_name}_{image_name}.png"
    plt.savefig(f"{save_path}{year}_{site_name}_{image_name}.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return file_name


def calc_stat(
    fig, ax, year: str, name: str, params: dict, global_attributes: dict
) -> tuple[list, list, np.ndarray]:
    bar_txt, stat = np.zeros((12, 2), np.float32), {}
    x_labels = [month_abbr[m_abbr] for m_abbr in range(1, 13)]

    if name == "data_availability":
        var_names = ["pointing_flag", "time_bnds", "quality_flag"]
        labels = ["single_pointing", "multiple_pointing"]
        stat = np.zeros((12, len(labels)), np.float32)
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):
                seqs = [(key, len(list(val))) for key, val in groupby(data["pointing_flag"] == 1)]
                seqs = np.array(
                    [
                        (key, sum(s[1] for s in seqs[:i]), len)
                        for i, (key, len) in enumerate(seqs)
                        if key == True
                    ]
                )
                bar_txt[month - 1, 0] = len(seqs)
                stat[month - 1, 0] = (
                    np.sum(np.abs(np.diff(data["time_bnds"][data["pointing_flag"] == 0, :])))
                    / 3600.0
                )
                if len(np.where(data["pointing_flag"] == 1)[0]) > 0:
                    stat[month - 1, 1] = (
                        np.sum(np.abs(np.diff(data["time_bnds"][data["pointing_flag"] == 1, :])))
                        / 3600.0
                    )
                bar_txt[month - 1, 1] = (
                    len(np.where(np.any(data["quality_flag"] == 0, axis=1))[0])
                    / len(data["quality_flag"])
                    * 100.0
                )

    if name == "quality_flag":
        var_names = ["quality_flag"]
        labels = np.array(
            [
                "missing_tb",
                "tb_below_threshold",
                "tb_above_threshold",
                "spectral_consistency_above_threshold",
                "receiver_sanity_failed",
                "rain_detected",
                "sun_moon_in_beam",
                "tb_offset_above_threshold",
            ]
        )
        bits = np.arange(len(labels))
        labels = labels[np.where(np.array(params["flag_status"]) == 0)[0]]
        bits = bits[np.where(np.array(params["flag_status"]) == 0)[0]]
        stat = np.zeros((12, len(labels)), np.float32)
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):
                for ibit, bit in enumerate(bits):
                    if len(np.where(np.any(data["quality_flag"] > 0, axis=1))[0]) > 0.:
                        stat[month - 1, ibit] = len(
                            np.where(np.any(isbit(data["quality_flag"], bit), axis=1))[0]
                        ) / len(np.where(np.any(data["quality_flag"] > 0, axis=1))[0])

    if name == "spectral_consistency":
        var_names = ["quality_flag", "time_bnds"]
        labels = np.array(
            [
                "spectral_consistency_above_threshold",
                "rain_detected",
            ]
        )
        data_ex = _get_data_ex(year, params, global_attributes)
        if bool(data_ex):
            stat = np.zeros((len(data_ex["frequency"][:]), 12, len(labels)), np.float32)
            bar_txt = [str(fname) for fname in np.array(data_ex["frequency"][:].data)]
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):
                for ifreq in np.arange(len(data_ex["frequency"][:])):
                    stat[ifreq, month - 1, 0] = (
                        ma.sum(
                            ma.abs(
                                ma.diff(
                                    data["time_bnds"][isbit(data["quality_flag"][:, ifreq], 3), :]
                                )
                            )
                        )
                        / 3600.0
                    )
                    if len(np.where(isbit(data["quality_flag"][:, ifreq], 3))[0]) > 0:
                        stat[ifreq, month - 1, 1] = (
                            len(
                                np.where(
                                    isbit(data["quality_flag"][:, ifreq], 3)
                                    & (isbit(data["quality_flag"][:, ifreq], 5))
                                )[0]
                            )
                            / len(np.where(isbit(data["quality_flag"][:, ifreq], 3))[0])
                            * 100.0
                        )

    if name == "receiver_temperature":
        var_names = ["t_rec", "time"]
        labels = ["Receiver 1", "Receiver 2"]
        stat_names = ["t_rec_mean", "t_rec_min", "t_rec_max"]
        stat = dict([(key, np.empty((0, 2))) for key in stat_names])
        stat["time"] = []
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):
                t_sta_df = pd.DataFrame(
                    data["t_rec"], columns=labels, index=pd.to_datetime(data["time"], unit="s")
                )
                df_res = t_sta_df.resample("D")[labels]
                stat["t_rec_max"] = np.vstack(
                    [stat["t_rec_max"], np.array(df_res.transform("max"))]
                )
                stat["t_rec_min"] = np.vstack(
                    [stat["t_rec_min"], np.array(df_res.transform("min"))]
                )
                stat["t_rec_mean"] = np.vstack(
                    [stat["t_rec_mean"], np.array(df_res.transform("mean"))]
                )
                stat["time"] = np.append(
                    stat["time"],
                    np.array(t_sta_df.index.month)
                    + np.array(t_sta_df.index.day) / np.array(t_sta_df.index.daysinmonth)
                    + np.array(t_sta_df.index.hour) / np.array(t_sta_df.index.daysinmonth) / 24.0,
                )

    if name == "receiver_stability":
        lab_pos = np.ones((12, 2), np.float32) * .003
        var_names = ["t_sta"]
        labels = ["Receiver 1", "Receiver 2"]
        ax2 = fig.add_subplot(111)
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):
                lab_pos[month - 1, 0] = np.nanpercentile(data["t_sta"][:, 0].filled(np.nan), 75)
                _boxplot(
                    ax,
                    data["t_sta"][:, 0],
                    labels=x_labels,
                    position=month,
                    color="sienna",
                    name=name,
                )
                lab_pos[month - 1, 1] = np.nanpercentile(data["t_sta"][:, 1].filled(np.nan), 75)
                _boxplot(
                    ax2,
                    data["t_sta"][:, 1],
                    labels=x_labels,
                    position=month,
                    color=_COLORS["shockred"],
                    name=name,
                )
        x_max = np.max(lab_pos) + .001
        ax.set_xlim([0., x_max])
        ax2.set_xlim([0., x_max])
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):        
                bar_txt = len(np.where(data["t_sta"][:, 0] > x_max)[0]) / len(data["t_sta"]) * 100
                if round(bar_txt, 2) > 0.0:
                    ax.text(
                        ax.get_xlim()[1],
                        month + 0.1,
                        str(round(bar_txt, 2)) + " %",
                        ha="center",
                        weight="bold",
                    )
                bar_txt = len(np.where(data["t_sta"][:, 1] > x_max)[0]) / len(data["t_sta"]) * 100
                if round(bar_txt, 2) > 0.0:
                    ax2.text(
                        ax2.get_xlim()[1],
                        month + 0.1,
                        str(round(bar_txt, 2)) + " %",
                        ha="center",
                        weight="bold",
                    )                    
        
        pos = ax.get_position().get_points()
        ax.set_position([0.06, 0.11, 0.31, pos[1, 1] - 0.12])
        ax2.set_position([0.42, 0.11, 0.31, pos[1, 1] - 0.12])
        ax2.set_title("Receiver 2 Stability (% above range)", fontsize=14)
        ax2.text(
            ax2.get_xlim()[1] + 0.0005,
            6.5,
            "Percentiles: \n10, 25, 50, 75, 90",
            bbox=dict(alpha=0.25, facecolor="w", boxstyle="round"),
        )

    if name == "ambient_target":
        var_names = ["t_amb", "time"]
        labels = ["Mean", "Difference"]
        stat_names = [
            "t_amb_mean",
            "t_amb_min",
            "t_amb_max",
            "t_amb_diff_mean",
            "t_amb_diff_min",
            "t_amb_diff_max",
        ]
        stat = dict([(key, np.empty((0))) for key in stat_names])
        stat["time"] = []
        for month in range(1, 13):
            data = _find_valid_fields(year, month, var_names, params, global_attributes)
            if bool(data):
                t_amb_df = pd.DataFrame(
                    ma.mean(data["t_amb"], axis=1),
                    columns=[labels[0]],
                    index=pd.to_datetime(data["time"], unit="s"),
                )
                t_amb_diff_df = pd.DataFrame(
                    ma.diff(data["t_amb"], axis=1),
                    columns=[labels[1]],
                    index=pd.to_datetime(data["time"], unit="s"),
                )
                df_res = t_amb_df.resample("D")[labels[0]]
                df_res_diff = t_amb_diff_df.resample("D")[labels[1]]

                stat["t_amb_max"] = np.concatenate(
                    [stat["t_amb_max"], np.array(df_res.transform("max"))]
                )
                stat["t_amb_min"] = np.concatenate(
                    [stat["t_amb_min"], np.array(df_res.transform("min"))]
                )
                stat["t_amb_mean"] = np.concatenate(
                    [stat["t_amb_mean"], np.array(df_res.transform("mean"))]
                )
                stat["t_amb_diff_max"] = np.concatenate(
                    [stat["t_amb_diff_max"], np.array(df_res_diff.transform("max"))]
                )
                stat["t_amb_diff_min"] = np.concatenate(
                    [stat["t_amb_diff_min"], np.array(df_res_diff.transform("min"))]
                )
                stat["t_amb_diff_mean"] = np.concatenate(
                    [stat["t_amb_diff_mean"], np.array(df_res_diff.transform("mean"))]
                )
                stat["time"] = np.append(
                    stat["time"],
                    np.array(t_amb_df.index.month)
                    + np.array(t_amb_df.index.day) / np.array(t_amb_df.index.daysinmonth)
                    + np.array(t_amb_df.index.hour) / np.array(t_amb_df.index.daysinmonth) / 24.0,
                )

    return stat, labels, bar_txt


def _find_valid_fields(
    year: str, month: int, names: list, params: dict, global_attributes: dict
) -> dict:
    """Returns valid field names and corresponding data."""
    valid_data = {}
    file_list = get_lev1_file_list(year, month, params, global_attributes)
    if len(file_list) > 0:
        for file in file_list:
            with netCDF4.Dataset(file) as nc:
                for var in names:
                    if var in nc.variables:
                        valid_data = append_data(valid_data, var, nc.variables[var])
    return valid_data


def _get_data_ex(year: str, params: dict, global_attributes: dict) -> netCDF4.Dataset:
    """Returns example data."""
    data_ex = {}
    for month in range(1, 13):
        file_list = get_lev1_file_list(year, month, params, global_attributes)
        if len(file_list) > 0:
            data_ex = netCDF4.Dataset(file_list[0])
    return data_ex


def get_lev1_file_list(year: str, month: int, params: dict, global_attributes: dict) -> list:
    """Returns file list of Level 1 files."""
    file_list = []
    data_out_l1 = params["data_out"] + "level1/"
    for day in daterange(year, month):
        file_name = (
            data_out_l1
            + day.strftime("%Y/%m/%d/")
            + "MWR_1C01_"
            + global_attributes["wigos_station_id"]
            + "_"
            + day.strftime("%Y%m%d")
            + ".nc"
        )
        if os.path.isfile(file_name):
            file_list = np.append(file_list, file_name)
    return file_list


def daterange(year: str, month: int):
    days = monthrange(int(year), month)
    start_date, end_date = date(int(year), month, 1), date(int(year), month, days[1])
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def _set_title(ax, field_name: str):
    """Sets title of plot."""
    ax.set_title(f"{ATTRIBUTES[field_name].name}", fontsize=14)


def _add_subtitle(fig, year: str, site_name: str):
    """Adds subtitle into figure."""
    site_name = site_name.replace("-", " ")
    text = f"{site_name}, {year}"
    fig.suptitle(
        text,
        fontsize=13,
        y=0.885,
        x=0.07,
        horizontalalignment="left",
        verticalalignment="bottom",
    )


def _plot_bar(
    ax, name: str, year: str, stat: np.ndarray, labels: list, bar_txt: np.ndarray, params: dict
):
    x_labels = [month_abbr[m_abbr] for m_abbr in range(1, 13)]
    colors = np.array(ATTRIBUTES[name].cbar)
    if name == "quality_flag":
        colors = colors[np.array(params["flag_status"]) == 0]
    width, N = 1, len(labels)
    x = np.arange(len(x_labels)) + 1
    for ind, l_name in enumerate(x_labels):
        x_loc = np.linspace(ind + 1 - width / 4, ind + 1 + width / 4, N, endpoint=False)
        if ind == N - 1:
            ax.bar(x_loc, stat[ind, :], width / N / 2, label=labels, color=colors, align="edge")
        else:
            ax.bar(x_loc, stat[ind, :], width / N / 2, label=None, color=colors, align="edge")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0.5, 12.5])
    ax.set_ylim(ATTRIBUTES[name].plot_range)
    ax.set_ylabel(ATTRIBUTES[name].ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if ATTRIBUTES[name].add_txt:
        for ind, txt in enumerate(bar_txt[:, 0]):
            if bar_txt[ind, 0] > 0.0:
                ax.text(
                    x[ind],
                    np.max(stat[ind, 1]) + 4,
                    str(int(txt)),
                    ha="left",
                    weight="bold",
                )
        for ind, txt in enumerate(bar_txt[:, 1]):
            if bar_txt[ind, 1] > 0.0:
                ax.text(
                    x[ind],
                    np.max(stat[ind, 0]) + 20,
                    str(round(txt, 1)) + " %",
                    ha="center",
                    weight="bold",
                )


def _plot_heat_map(ax, name: str, year: str, stat: np.ndarray, labels: list, bar_txt: np.ndarray):
    masked_array = np.ma.masked_where(stat == 0.0, stat)
    x_labels = [month_abbr[m_abbr] for m_abbr in range(1, 13)]
    cmap = cm.summer
    cmap.set_bad(color=_COLORS["gray"])
    bbox = ax.get_position()
    im, cbar = heatmap(
        masked_array[:, :, 0],
        bar_txt,
        x_labels,
        ax=ax,
        cmap=cmap,
        cbarlabel=labels[0] + " (hours)",
        aspect="auto",
        vmin=np.min(stat[stat > 0]),
        origin="lower",
    )
    ax.set_position(bbox)
    texts = annotate_heatmap(
        im, stat[:, :, 1], valfmt="{x:.1f} %", threshold=stat[stat[:, :, 0] > 0, 0].max() / 2.0
    )
    ax.set_ylabel(ATTRIBUTES[name].ylabel)


def _plot_lines_rec(ax, name: str, stat: np.ndarray, labels: list):
    x_labels = [month_abbr[m_abbr] for m_abbr in range(1, 13)]
    gaps = np.where(np.diff(stat["time"], append=stat["time"][-1] + 0.01) > 2.0 / 30.0)[0]
    t_rec_mean = ma.array(stat["t_rec_mean"])

    if len(gaps) > 0:
        t_rec_mean[gaps, :] = ma.masked

    ax.set_xlim([1.0, 13.0])
    ax.set_xticks(np.arange(12) + 1)
    ax.set_xticklabels(x_labels)

    ax.plot(stat["time"], t_rec_mean[:, 0], color="sienna", label="Receiver 1")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2 = ax.twinx()
    ax2.plot(
        stat["time"],
        t_rec_mean[:, 1],
        color=_COLORS["shockred"],
        label="Receiver 2",
    )
    vmin1, vmax1 = (
        np.nanmin(stat["t_rec_min"][:, 0]) - 0.05,
        np.nanmax(stat["t_rec_max"][:, 0]) + 0.05,
    )
    vmin2, vmax2 = (
        np.nanmin(stat["t_rec_min"][:, 1]) - 0.05,
        np.nanmax(stat["t_rec_max"][:, 1]) + 0.05,
    )
    if vmax1 - vmin1 > vmax2 - vmin2:
        ax.set_ylim([vmin1, vmax1])
        ax2.set_ylim(
            [
                vmin2 - ((vmax1 - vmin1) - (vmax2 - vmin2)) / 2,
                vmax2 + ((vmax1 - vmin1) - (vmax2 - vmin2)) / 2,
            ]
        )
    else:
        ax.set_ylim(
            [
                vmin1 - ((vmax2 - vmin2) - (vmax1 - vmin1)) / 2,
                vmax1 + ((vmax2 - vmin2) - (vmax1 - vmin1)) / 2,
            ]
        )
        ax2.set_ylim([vmin2, vmax2])

    ax.fill_between(
        stat["time"],
        stat["t_rec_min"][:, 0],
        stat["t_rec_max"][:, 0],
        facecolor="sienna",
        alpha=0.7,
        where=(np.diff(stat["time"], append=stat["time"][-1] + 0.01) < 2.0 / 30.0),
        interpolate=False,
    )

    ax2.fill_between(
        stat["time"],
        stat["t_rec_min"][:, 1],
        stat["t_rec_max"][:, 1],
        facecolor=_COLORS["shockred"],
        alpha=0.4,
        where=(np.diff(stat["time"], append=stat["time"][-1] + 0.01) < 2.0 / 30.0),
        interpolate=False,
    )

    ax.set_ylabel("Receiver 1 daily mean and range (K)")
    ax2.set_ylabel("Receiver 2 daily mean and range (K)")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax2.legend(
        lines + lines2, labels + labels2, loc="center left", bbox_to_anchor=(1.08, 0.5)
    )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)


def _plot_lines_amb(ax, name: str, stat: np.ndarray, labels: list):
    x_labels = [month_abbr[m_abbr] for m_abbr in range(1, 13)]
    gaps = np.where(np.diff(stat["time"], append=stat["time"][-1] + 0.01) > 2.0 / 30.0)[0]
    t_amb_mean = ma.array(stat["t_amb_mean"])
    t_amb_diff = ma.array(stat["t_amb_diff_mean"])

    if len(gaps) > 0:
        t_amb_mean[gaps] = ma.masked
        t_amb_diff[gaps] = ma.masked

    ax.set_xlim([1.0, 13.0])
    ax.set_xticks(np.arange(12) + 1)
    ax.set_xticklabels(x_labels)

    ax.plot(stat["time"], t_amb_mean, color="darkblue", label=labels[0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2 = ax.twinx()
    ax2.plot(
        stat["time"],
        t_amb_diff,
        color=_COLORS["darkgray"],
        label=labels[1],
    )
    vmin1, vmax1 = np.nanmin(stat["t_amb_min"]) - 0.05, np.nanmax(stat["t_amb_max"]) + 0.05
    vmin2, vmax2 = (
        np.nanmin(stat["t_amb_diff_min"]) - 0.05,
        np.nanmax(stat["t_amb_diff_max"]) + 0.05,
    )

    ax.fill_between(
        stat["time"],
        stat["t_amb_min"],
        stat["t_amb_max"],
        facecolor="darkblue",
        alpha=0.7,
        where=(np.diff(stat["time"], append=stat["time"][-1] + 0.01) < 2.0 / 30.0),
        interpolate=False,
    )

    ax2.fill_between(
        stat["time"],
        stat["t_amb_diff_min"],
        stat["t_amb_diff_max"],
        facecolor=_COLORS["darkgray"],
        alpha=0.4,
        where=(np.diff(stat["time"], append=stat["time"][-1] + 0.01) < 2.0 / 30.0),
        interpolate=False,
    )

    ax.set_ylabel(ATTRIBUTES[name].ylabel)
    ax2.set_ylim([0., np.min([.3, ma.max(stat["t_amb_diff_max"]) + .01])])
    ax2.set_ylabel("Daily mean and range of sensor difference (K)")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax2.legend(
        lines + lines2, labels + labels2, loc="center left", bbox_to_anchor=(1.08, 0.5)
    )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)


def _boxplot(ax, data, labels=None, position=0, color=None, name=None):
    ax.boxplot(
        data,
        vert=False,
        positions=range(position, position + 1),
        manage_ticks=False,
        whis=(10, 90),
        sym="",
        widths=0.7,
        patch_artist=True,
        medianprops=dict(color=color, linewidth=2),
        boxprops=dict(facecolor="w"),
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(ATTRIBUTES[name].ylabel)
    ax.set_ylim([0.5, 12.5])
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(labels)
