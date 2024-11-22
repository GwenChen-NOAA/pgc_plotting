#!/usr/bin/env python3
# =============================================================================
#
# NAME: plotter.py
# CONTRIBUTOR(S): Marcel Caron, marcel.caron@noaa.gov, NOAA/NWS/NCEP/EMC-VPPPGB
# PURPOSE: Plotting specifications for CAM plotting scripts
#
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import numpy as np
import pandas as pd
import math
import os
import sys
SETTINGS_DIR = os.environ['USH_DIR']
sys.path.insert(0, os.path.abspath(SETTINGS_DIR))
import plot_util

class Plotter():
    def __init__(self, font_weight='bold',  axis_title_weight='bold',  
                axis_title_size=15,         axis_offset=False,
                axis_title_pad=10,          axis_label_weight='bold',  
                axis_label_size=14,         axis_label_pad=10,
                axis_fontsize=16,           clabel_font_size=10,
                xtick_label_size=16,        xtick_major_pad=10,        
                ytick_label_size=16,        ytick_major_pad=10, 
                fig_subplot_right=.95,      fig_subplot_left=.1,      
                fig_subplot_top=.87,        fig_subplot_bottom=.2,
                legend_handle_text_pad=.4,  legend_handle_length=3., 
                legend_border_axis_pad=.5,  legend_col_space=3.,
                legend_frame_on=True,       fig_size=(16.,8.),        
                legend_bbox=(0,1),          legend_font_size=12, 
                legend_loc='upper center',  legend_ncol=4,        
                lines_line_width=1.,
                title_loc='center',         title_color='black'):
        self.font_weight = font_weight
        self.axis_title_weight = axis_title_weight
        self.axis_title_size = axis_title_size
        self.axis_title_pad = axis_title_pad
        self.axis_offset = axis_offset
        self.axis_label_weight = axis_label_weight
        self.axis_label_size = axis_label_size
        self.axis_label_pad = axis_label_pad
        self.axis_fontsize = axis_fontsize
        self.clabel_font_size = clabel_font_size
        self.xtick_label_size = xtick_label_size
        self.xtick_major_pad = xtick_major_pad
        self.ytick_label_size = ytick_label_size
        self.ytick_major_pad = ytick_major_pad
        self.fig_size = fig_size
        self.fig_subplot_right = fig_subplot_right
        self.fig_subplot_left = fig_subplot_left
        self.fig_subplot_top = fig_subplot_top
        self.fig_subplot_bottom = fig_subplot_bottom
        self.legend_handle_text_pad = legend_handle_text_pad
        self.legend_handle_length = legend_handle_length
        self.legend_border_axis_pad = legend_border_axis_pad
        self.legend_col_space = legend_col_space
        self.legend_frame_on = legend_frame_on
        self.legend_bbox = legend_bbox
        self.legend_font_size = legend_font_size
        self.legend_loc = legend_loc
        self.legend_ncol = legend_ncol
        self.lines_line_width = lines_line_width
        self.title_loc = title_loc
        self.title_color = title_color
        self.f = lambda m,c,ls,lw,ms,mec: plt.plot(
            [], [], marker=m, mec=mec, mew=2., c=c, ls=ls, lw=lw, ms=ms
        )[0]

    def set_up_plots(self):
        plt.rcParams['axes.formatter.useoffset'] = self.axis_offset
        plt.rcParams['axes.labelpad'] = self.axis_label_pad
        plt.rcParams['axes.labelsize'] = self.axis_label_size
        plt.rcParams['axes.labelweight'] = self.axis_label_weight
        plt.rcParams['axes.titlecolor'] = self.title_color
        plt.rcParams['axes.titlelocation'] = self.title_loc
        plt.rcParams['axes.titlepad'] = self.axis_title_pad
        plt.rcParams['axes.titlesize'] = self.axis_title_size
        plt.rcParams['axes.titleweight'] = self.axis_title_weight
        plt.rcParams['figure.figsize'] = self.fig_size
        plt.rcParams['figure.subplot.bottom'] = self.fig_subplot_bottom
        plt.rcParams['figure.subplot.left'] = self.fig_subplot_left
        plt.rcParams['figure.subplot.right'] = self.fig_subplot_right
        plt.rcParams['figure.subplot.top'] = self.fig_subplot_top
        plt.rcParams['font.size'] = self.axis_fontsize
        plt.rcParams['font.weight'] = self.font_weight
        plt.rcParams['legend.handletextpad'] = self.legend_handle_text_pad
        plt.rcParams['legend.handlelength'] = self.legend_handle_length
        plt.rcParams['legend.borderaxespad'] = self.legend_border_axis_pad
        plt.rcParams['legend.columnspacing'] = self.legend_col_space
        plt.rcParams['legend.frameon'] = self.legend_frame_on
        plt.rcParams['legend.fontsize'] = self.legend_font_size
        plt.rcParams['legend.loc'] = self.legend_loc
        plt.rcParams['lines.linewidth'] = self.lines_line_width
        plt.rcParams['xtick.labelsize'] = self.xtick_label_size
        plt.rcParams['xtick.major.pad'] = self.xtick_major_pad
        plt.rcParams['ytick.labelsize'] = self.ytick_label_size
        plt.rcParams['ytick.major.pad'] = self.ytick_major_pad
      
    def get_plots(self, num):
        fig, ax = plt.subplots(1, 1, figsize=self.fig_size, num=num)
        return fig, ax

    def get_error_boxes(self, xdata, ydata, xerror, yerror, fc='None', 
                       ec='black', lw=1., ls='solid', alpha=0.75):
        errorboxes = []
        xerror = np.array(xerror)
        yerror = np.array(yerror)
        for xc, yc, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
            rect = Rectangle((xc+xe[0], yc+ye[0]), np.diff(xe), np.diff(ye))
            errorboxes.append(rect)
        pc = PatchCollection(
            errorboxes, facecolor=fc, alpha=alpha, edgecolor=ec, linewidth=lw, 
            linestyle=ls
        )
        return pc

    def get_error_brackets(self, xdata, ydata, xerror, yerror, fc='None', 
                          ec='black', lw=1., alpha=0.75):
        errorbrackets = []
        verts = []
        codes = []
        xerror = np.array(xerror)
        yerror = np.array(yerror)
        for xc, yc, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
            verts += [
                (xc+xe[1], yc+ye[0]),
                (xc+xe[0], yc+ye[0]),
                (xc+xe[0], yc+ye[1]),
                (xc+xe[1], yc+ye[1])
            ]
            codes += [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
            ]
        path = Path(verts, codes)
        pp = PathPatch(
            path, facecolor=fc, alpha=alpha, edgecolor=ec, linewidth=lw
        )
        return pp

    def get_logo_location(self, position, x_figsize, y_figsize, dpi):
        """! Get locations for the logos

            Args:
                position  - side of image (string, "left" or "right")
                x_figsize - image size in x direction (float)
                y_figsize - image size in y_direction (float)
                dpi       - image dots per inch (float)

            Returns:
                x_loc - logo position in x direction (float)
                y_loc - logo position in y_direction (float)
                alpha - alpha value (float)
        """
        alpha = 0.5
        if x_figsize == 8 and y_figsize == 6:
            if position == 'left':
                x_loc = x_figsize * dpi * 0.0
                y_loc = y_figsize * dpi * 0.858
            elif position == 'right':
                x_loc = x_figsize * dpi * 0.9
                y_loc = y_figsize * dpi * 0.858
        elif x_figsize == 16 and y_figsize == 8:
            if position == 'left':
                x_loc = x_figsize * dpi * 0.0
                y_loc = y_figsize * dpi * 0.89
            elif position == 'right':
                x_loc = x_figsize * dpi * 0.948
                y_loc = y_figsize * dpi * 0.89
        elif x_figsize == 16 and y_figsize == 16:
            if position == 'left':
                x_loc = x_figsize * dpi * 0.0
                y_loc = y_figsize * dpi * 0.945
            elif position == 'right':
                x_loc = x_figsize * dpi * 0.948
                y_loc = y_figsize * dpi * 0.945
        else:
            if position == 'left':
                x_loc = x_figsize * dpi * 0.0
                y_loc = y_figsize * dpi * 0.95
            elif position == 'right':
                x_loc = x_figsize * dpi * 0.948
                y_loc = y_figsize * dpi * 0.95
        return x_loc, y_loc, alpha
    
    def plot_by_lead(self, fig, pivot_metric1, pivot_metric2, pivot_counts,
                      pivot_ci_lower1, pivot_ci_upper1, pivot_ci_lower2, 
                      pivot_ci_upper2, y_min, y_max, y_min_limit, y_max_limit, 
                      x_vals1, x_vals2, metric1_name, metric2_name, flead, 
                      model_list, model_colors, setting_dicts, 
                      confidence_intervals=False, y_lim_lock=True,
                      display_averages=False, running_mean=''):
        if metric2_name is not None:
            handles = [
                self.f('', 'black', line_setting, 5., 0, 'white')
                for line_setting in ['solid','dashed']
            ]
            labels = [
                str(metric_name).upper()
                for metric_name in [metric1_name, metric2_name]
            ]
        else:
            handles = []
            labels = []
        n_mods = 0
        for l in range(len(flead)):
            if flead[l] >= 24 and int(flead[l])%24 in [0, 6, 12, 18]:
                if int(flead[l])%24 == 0:
                    use_flead = str(int(flead[l]/24.))
                else:
                    use_flead = str(round(flead[l]/24., 2))
                flead_plot_name = f"Day {use_flead}"
            else:
                flead_plot_name = f"F{flead[l]:03d}"
            if flead[l] not in pivot_metric1:
                continue
            if not model_list:
                model_plot_name = ""
            elif len(model_list) == 1:
                if model_list[0] in model_colors.model_alias:
                    model_plot_name = model_colors.model_alias[
                        model_list[0]
                    ]['plot_name']
                else:
                    model_plot_name = model_list[0]
            else:
                model_plot_name = ""
                for m in range(len(model_list[:-1])):
                    if model_list[m] in model_colors.model_alias:
                        model_plot_name+=(
                            ', '+model_colors.model_alias[
                                model_list[m]
                            ]['plot_name']
                        )
                    else:
                        model_plot_name+=(
                            ', '+model_list[m]
                        )
                if model_list[-1] in model_colors.model_alias:
                    model_plot_name+=(
                        ', and '+model_colors.model_alias[
                            model_list[-1]
                        ]['plot_name']
                    )
                else:
                    model_plot_name+=(
                        ', and '+model_list[-1]
                    )
            y_vals_metric1 = pivot_metric1[flead[l]].values
            y_vals_metric1_mean = np.nanmean(y_vals_metric1)
            if metric2_name is not None:
                y_vals_metric2 = pivot_metric2[flead[l]].values
                y_vals_metric2_mean = np.nanmean(y_vals_metric2)
            if confidence_intervals:
                y_vals_ci_lower1 = pivot_ci_lower1[
                    flead[l]
                ].values
                y_vals_ci_upper1 = pivot_ci_upper1[
                    flead[l]
                ].values
                if metric2_name is not None:
                    y_vals_ci_lower2 = pivot_ci_lower2[
                        flead[l]
                    ].values
                    y_vals_ci_upper2 = pivot_ci_upper2[
                        flead[l]
                    ].values
            if not y_lim_lock:
                if metric2_name is not None:
                    y_vals_both_metrics = np.concatenate((y_vals_metric1, y_vals_metric2))
                    if np.any(y_vals_both_metrics != np.inf):
                        y_vals_metric_min = np.nanmin(y_vals_both_metrics[y_vals_both_metrics != np.inf])
                        y_vals_metric_max = np.nanmax(y_vals_both_metrics[y_vals_both_metrics != np.inf])
                    else:
                        y_vals_metric_min = np.nanmin(y_vals_both_metrics)
                        y_vals_metric_max = np.nanmax(y_vals_both_metrics)
                else:
                    if np.any(y_vals_metric1 != np.inf):
                        y_vals_metric_min = np.nanmin(y_vals_metric1[y_vals_metric1 != np.inf])
                        y_vals_metric_max = np.nanmax(y_vals_metric1[y_vals_metric1 != np.inf])
                    else:
                        y_vals_metric_min = np.nanmin(y_vals_metric1)
                        y_vals_metric_max = np.nanmax(y_vals_metric1)
                if n_mods == 0:
                    y_mod_min = y_vals_metric_min
                    y_mod_max = y_vals_metric_max
                    counts = pivot_counts[flead[l]].values
                    n_mods+=1
                else:
                    if math.isinf(y_mod_min):
                        y_mod_min = y_vals_metric_min
                    else:
                        y_mod_min = np.nanmin([y_mod_min, y_vals_metric_min])
                    if math.isinf(y_mod_max):
                        y_mod_max = y_vals_metric_max
                    else:
                        y_mod_max = np.nanmax([y_mod_max, y_vals_metric_max])
                if (y_vals_metric_min > y_min_limit 
                        and y_vals_metric_min <= y_mod_min):
                    y_min = y_vals_metric_min
                if (y_vals_metric_max < y_max_limit 
                        and y_vals_metric_max >= y_mod_max):
                    y_max = y_vals_metric_max
            if np.abs(y_vals_metric1_mean) < 1E4:
                metric1_mean_fmt_string = f'{y_vals_metric1_mean:.2f}'
            else:
                metric1_mean_fmt_string = f'{y_vals_metric1_mean:.2E}'
            if running_mean:
                alpha=.25
            else:
                alpha=1.0
            plt.plot(
                x_vals1.tolist(), y_vals_metric1, 
                marker=setting_dicts[l]['marker'], 
                c=setting_dicts[l]['color'], mew=2., mec='white', 
                figure=fig, ms=setting_dicts[l]['markersize'], ls='solid', 
                lw=setting_dicts[l]['linewidth'],alpha=alpha
            )
            if running_mean:
                y_vals_rolling1 = plot_util.get_rolling_mean(y_vals_metric1, running_mean)
                plt.plot(
                    x_vals1.tolist(), y_vals_rolling1.tolist(),
                    marker=None, c=setting_dicts[l]['color'], figure=fig, 
                    ms=0., ls='solid', lw=setting_dicts[l]['linewidth']*2,
                    alpha=1.0
                )
            if metric2_name is not None:
                if np.abs(y_vals_metric2_mean) < 1E4:
                    metric2_mean_fmt_string = f'{y_vals_metric2_mean:.2f}'
                else:
                    metric2_mean_fmt_string = f'{y_vals_metric2_mean:.2E}'
                plt.plot(
                    x_vals2.tolist(), y_vals_metric2, 
                    marker=setting_dicts[l]['marker'], 
                    c=setting_dicts[l]['color'], mew=2., mec='white', 
                    figure=fig, ms=setting_dicts[l]['markersize'], 
                    ls='dashed', lw=setting_dicts[l]['linewidth'], alpha=alpha
                )
                if running_mean:
                    y_vals_rolling2 = plot_util.get_rolling_mean(y_vals_metric2, running_mean)
                    plt.plot(
                        x_vals2.tolist(), y_vals_rolling2.tolist(),
                        marker=None, c=setting_dicts[l]['color'], figure=fig, 
                        ms=0., ls='dashed', lw=setting_dicts[l]['linewidth']*2,
                        alpha=1.0
                    )
            if confidence_intervals:
                plt.errorbar(
                    x_vals1.tolist(), y_vals_metric1,
                    yerr=[np.abs(y_vals_ci_lower1), y_vals_ci_upper1],
                    fmt='none', ecolor=setting_dicts[l]['color'],
                    elinewidth=setting_dicts[l]['linewidth'],
                    capsize=10., capthick=setting_dicts[l]['linewidth'],
                    alpha=alpha*.70, zorder=0
                )
                if metric2_name is not None:
                    plt.errorbar(
                        x_vals2.tolist(), y_vals_metric2,
                        yerr=[np.abs(y_vals_ci_lower2), y_vals_ci_upper2],
                        fmt='none', ecolor=setting_dicts[l]['color'],
                        elinewidth=setting_dicts[l]['linewidth'],
                        capsize=10., capthick=setting_dicts[l]['linewidth'],
                        alpha=alpha*.70, zorder=0
                    )
            handles+=[
                self.f(
                    setting_dicts[l]['marker'], setting_dicts[l]['color'],
                    'solid', setting_dicts[l]['linewidth'], 
                    setting_dicts[l]['markersize'], 'white'
                )
            ]
            if display_averages:
                if metric2_name is not None:
                    labels+=[
                        f'{flead_plot_name} ({metric1_mean_fmt_string},'
                        + f' {metric2_mean_fmt_string})'
                    ]
                else:
                    labels+=[
                        f'{flead_plot_name} ({metric1_mean_fmt_string})'
                    ]
            else:
                labels+=[f'{flead_plot_name}']

        return (fig, y_min, y_max, handles, labels)
 
    def plot_by_metric(self, fig, pivot_metric1, pivot_metric2, pivot_counts,
                      pivot_ci_lower1, pivot_ci_upper1, pivot_ci_lower2, 
                      pivot_ci_upper2, y_min, y_max, y_min_limit, y_max_limit, 
                      x_vals1, x_vals2, metric1_name, metric2_name, flead, 
                      model_list, model_colors, setting_dicts, 
                      confidence_intervals=False, y_lim_lock=True,
                      display_averages=False, running_mean='', target_vals=[0.5]):
        pivot_interpolated1 = plot_util.get_pivot_table_by_val(pivot_metric1, target_vals)
        pivot_counts = pivot_interpolated1.copy()
        pivot_counts[:] = np.nan
        if confidence_intervals:
            pivot_ci_lower1 = plot_util.get_pivot_table_by_val(pivot_ci_lower1, target_vals)
            pivot_ci_upper1 = plot_util.get_pivot_table_by_val(pivot_ci_upper1, target_vals)
        if metric2_name is not None:
            pivot_interpolated2 = plot_util.get_pivot_table_by_val(pivot_metric2, target_vals)
            if confidence_intervals:
                pivot_ci_lower2 = plot_util.get_pivot_table_by_val(pivot_ci_lower2, target_vals)
                pivot_ci_upper2 = plot_util.get_pivot_table_by_val(pivot_ci_upper2, target_vals)

        if metric2_name is not None:
            handles = [
                self.f('', 'black', line_setting, 5., 0, 'white')
                for line_setting in ['solid','dashed']
            ]
            labels = [
                str(metric_name).upper()
                for metric_name in [metric1_name, metric2_name]
            ]
        else:
            handles = []
            labels = []
        n_mods = 0
        val_categories = np.array([
            [np.power(10.,y), 2.*np.power(10.,y)]
            for y in [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        ]).flatten()
        round_to_nearest_categories = val_categories/20.
        round_to_nearest = round_to_nearest_categories[
            np.digitize(
                np.max(target_vals)-np.min(target_vals), 
                val_categories[:-1]
            )
        ]
        if round_to_nearest < 1.:
            val_precision_scale = 100/round_to_nearest
        else:
            val_precision_scale = 1.
        use_vals = [
            target_val*val_precision_scale for target_val in target_vals
        ]
        use_vals = np.divide(use_vals, val_precision_scale)
        for v in range(len(target_vals)):
            
            if metric2_name is not None:
                val_plot_name = f"={use_vals[v]}"
            else:
                val_plot_name = f"{str(metric1_name).upper()}={use_vals[v]}"
            if target_vals[v] not in pivot_interpolated1:
                continue
            if not model_list:
                model_plot_name = ""
            elif len(model_list) == 1:
                if model_list[0] in model_colors.model_alias:
                    model_plot_name = model_colors.model_alias[
                        model_list[0]
                    ]['plot_name']
                else:
                    model_plot_name = model_list[0]
            else:
                model_plot_name = ""
                for m in range(len(model_list[:-1])):
                    if model_list[m] in model_colors.model_alias:
                        model_plot_name+=(
                            ', '+model_colors.model_alias[
                                model_list[m]
                            ]['plot_name']
                        )
                    else:
                        model_plot_name+=(
                            ', '+model_list[m]
                        )
                if model_list[-1] in model_colors.model_alias:
                    model_plot_name+=(
                        ', and '+model_colors.model_alias[
                            model_list[-1]
                        ]['plot_name']
                    )
                else:
                    model_plot_name+=(
                        ', and '+model_list[-1]
                    )
            y_vals_metric1 = np.array(
                pivot_interpolated1[target_vals[v]].values.tolist()
            )
            y_vals_metric1_mean = np.nanmean(y_vals_metric1)
            if metric2_name is not None:
                y_vals_metric2 = np.array(
                    pivot_interpolated2[target_vals[v]].values.tolist()
                )
                y_vals_metric2_mean = np.nanmean(y_vals_metric2)
            if confidence_intervals:
                y_vals_ci_lower1 = pivot_ci_lower1[
                    target_vals[v]
                ].values.tolist()
                y_vals_ci_upper1 = pivot_ci_upper1[
                    target_vals[v]
                ].values.tolist()
                if metric2_name is not None:
                    y_vals_ci_lower2 = pivot_ci_lower2[
                        target_vals[v]
                    ].values.tolist()
                    y_vals_ci_upper2 = pivot_ci_upper2[
                        target_vals[v]
                    ].values.tolist()
            if not y_lim_lock:
                if metric2_name is not None:
                    y_vals_both_metrics = np.concatenate((y_vals_metric1, y_vals_metric2))
                    if np.any(y_vals_both_metrics != np.inf):
                        y_vals_metric_min = np.nanmin(y_vals_both_metrics[y_vals_both_metrics != np.inf])
                        y_vals_metric_max = np.nanmax(y_vals_both_metrics[y_vals_both_metrics != np.inf])
                    else:
                        y_vals_metric_min = np.nanmin(y_vals_both_metrics)
                        y_vals_metric_max = np.nanmax(y_vals_both_metrics)
                else:
                    if np.any(y_vals_metric1 != np.inf):
                        y_vals_metric_min = np.nanmin(y_vals_metric1[y_vals_metric1 != np.inf])
                        y_vals_metric_max = np.nanmax(y_vals_metric1[y_vals_metric1 != np.inf])
                    else:
                        y_vals_metric_min = np.nanmin(y_vals_metric1)
                        y_vals_metric_max = np.nanmax(y_vals_metric1)
                if n_mods == 0:
                    y_mod_min = y_vals_metric_min
                    y_mod_max = y_vals_metric_max
                    counts = pivot_counts[target_vals[v]].values
                    n_mods+=1
                else:
                    if math.isinf(y_mod_min):
                        y_mod_min = y_vals_metric_min
                    else:
                        y_mod_min = np.nanmin([y_mod_min, y_vals_metric_min])
                    if math.isinf(y_mod_max):
                        y_mod_max = y_vals_metric_max
                    else:
                        y_mod_max = np.nanmax([y_mod_max, y_vals_metric_max])
                if (y_vals_metric_min > y_min_limit 
                        and y_vals_metric_min <= y_mod_min):
                    y_min = y_vals_metric_min
                if (y_vals_metric_max < y_max_limit 
                        and y_vals_metric_max >= y_mod_max):
                    y_max = y_vals_metric_max
            if np.abs(y_vals_metric1_mean) < 1E4:
                metric1_mean_fmt_string = f'{y_vals_metric1_mean:.2f}'
            else:
                metric1_mean_fmt_string = f'{y_vals_metric1_mean:.2E}'
            if running_mean:
                alpha=.25
            else:
                alpha=1.0
            plt.plot(
                x_vals1.tolist(), y_vals_metric1, 
                marker=setting_dicts[v]['marker'], 
                c=setting_dicts[v]['color'], mew=2., mec='white', 
                figure=fig, ms=setting_dicts[v]['markersize'], ls='solid', 
                lw=setting_dicts[v]['linewidth'], alpha=alpha
            )
            if running_mean:
                y_vals_rolling1 = plot_util.get_rolling_mean(y_vals_metric1, running_mean)
                plt.plot(
                    x_vals1.tolist(), y_vals_rolling1.tolist(),
                    marker=None, c=setting_dicts[v]['color'], figure=fig, 
                    ms=0., ls='solid', lw=setting_dicts[v]['linewidth']*2,
                    alpha=1.0
                )
            if metric2_name is not None:
                if np.abs(y_vals_metric2_mean) < 1E4:
                    metric2_mean_fmt_string = f'{y_vals_metric2_mean:.2f}'
                else:
                    metric2_mean_fmt_string = f'{y_vals_metric2_mean:.2E}'
                plt.plot(
                    x_vals2.tolist(), y_vals_metric2, 
                    marker=setting_dicts[v]['marker'], 
                    c=setting_dicts[v]['color'], mew=2., mec='white', 
                    figure=fig, ms=setting_dicts[v]['markersize'], 
                    ls='dashed', lw=setting_dicts[v]['linewidth'], alpha=alpha
                )
                if running_mean:
                    y_vals_rolling2 = plot_util.get_rolling_mean(
                        y_vals_metric2, running_mean
                    )
                    plt.plot(
                        x_vals2.tolist(), y_vals_rolling2.tolist(),
                        marker=None, c=setting_dicts[v]['color'], figure=fig, 
                        ms=0., ls='dashed', lw=setting_dicts[v]['linewidth'],
                        alpha=1.0
                    )
            if confidence_intervals:
                plt.errorbar(
                    x_vals1.tolist(), y_vals_metric1,
                    yerr=[np.abs(y_vals_ci_lower1), y_vals_ci_upper1],
                    fmt='none', ecolor=setting_dicts[v]['color'],
                    elinewidth=setting_dicts[v]['linewidth'],
                    capsize=10., capthick=setting_dicts[v]['linewidth'],
                    alpha=.70*alpha, zorder=0
                )
                if metric2_name is not None:
                    plt.errorbar(
                        x_vals2.tolist(), y_vals_metric2,
                        yerr=[np.abs(y_vals_ci_lower2), y_vals_ci_upper2],
                        fmt='none', ecolor=setting_dicts[v]['color'],
                        elinewidth=setting_dicts[v]['linewidth'],
                        capsize=10., capthick=setting_dicts[v]['linewidth'],
                        alpha=.70*alpha, zorder=0
                    )
            handles+=[
                self.f(
                    setting_dicts[v]['marker'], setting_dicts[v]['color'],
                    'solid', setting_dicts[v]['linewidth'], 
                    setting_dicts[v]['markersize'], 'white'
                )
            ]
            if display_averages:
                if metric2_name is not None:
                    labels+=[
                        f'{val_plot_name} ({metric1_mean_fmt_string},'
                        + f' {metric2_mean_fmt_string})'
                    ]
                else:
                    labels+=[
                        f'{val_plot_name} ({metric1_mean_fmt_string})'
                    ]
            else:
                labels+=[f'{val_plot_name}']

        return (fig, y_min, y_max, handles, labels)

    def plot_by_model(self, fig, pivot_metric1, pivot_metric2, pivot_counts,
                      pivot_ci_lower1, pivot_ci_upper1, pivot_ci_lower2, 
                      pivot_ci_upper2, y_min, y_max, y_min_limit, y_max_limit, 
                      x_vals1, x_vals2, metric1_name, metric2_name, 
                      model_list, model_colors, mod_setting_dicts, 
                      confidence_intervals=False, y_lim_lock=True,
                      display_averages=False):
        plot_reference = [False, False]
        ref_metrics = ['OBAR']
        if str(metric1_name).upper() in ref_metrics:
            plot_reference[0] = True
            pivot_reference1 = pivot_metric1
            reference1 = pivot_reference1.mean(axis=1)
            if confidence_intervals:
                reference_ci_lower1 = pivot_ci_lower1.mean(axis=1)
                reference_ci_upper1 = pivot_ci_upper1.mean(axis=1)
            if not np.any((pivot_reference1.T/reference1).T == 1.):
                logger.warning(
                    f"{str(metric1_name).upper()} is requested, but the value "
                    + f"varies from model to model. "
                    + f"Will plot an individual line for each model. If a "
                    + f"single reference line is preferred, set the "
                    + f"sample_equalization toggle in ush/settings.py to 'True', "
                    + f"and check in the log file if sample equalization "
                    + f"completed successfully."
                )
                plot_reference[0] = False
        if metric2_name is not None and str(metric2_name).upper() in ref_metrics:
            plot_reference[1] = True
            pivot_reference2 = pivot_metric2
            reference2 = pivot_reference2.mean(axis=1)
            if confidence_intervals:
                reference_ci_lower2 = pivot_ci_lower2.mean(axis=1)
                reference_ci_upper2 = pivot_ci_upper2.mean(axis=1)
            if not np.any((pivot_reference2.T/reference2).T == 1.):
                logger.warning(
                    f"{str(metric2_name).upper()} is requested, but the value "
                    + f"varies from model to model. "
                    + f"Will plot an individual line for each model. If a "
                    + f"single reference line is preferred, set the "
                    + f"sample_equalization toggle in ush/settings.py to 'True', "
                    + f"and check in the log file if sample equalization "
                    + f"completed successfully."
                )
                plot_reference[1] = False
        if np.any(plot_reference):
            plotted_reference = [False, False]
            if confidence_intervals:
                plotted_reference_CIs = [False, False]
        if metric2_name is not None:
            if np.any(plot_reference):
                ref_color_dict = model_colors.get_color_dict('obs')
                handles = []
                labels = []
                line_settings = ['solid','dashed']
                metric_names = [metric1_name, metric2_name]
                for p, rbool in enumerate(plot_reference):
                    if rbool:
                        handles += [
                            self.f('', ref_color_dict['color'], line_settings[p], 5., 0, 'white')
                        ]
                    else:
                        handles += [
                            self.f('', 'black', line_settings[p], 5., 0, 'white')
                        ]
                    labels += [
                        str(metric_names[p]).upper()
                    ]
            else:
                handles = [
                    self.f('', 'black', line_setting, 5., 0, 'white')
                    for line_setting in ['solid','dashed']
                ]
                labels = [
                    str(metric_name).upper()
                    for metric_name in [metric1_name, metric2_name]
                ]
        else:
            handles = []
            labels = []
        n_mods = 0
        for m in range(len(mod_setting_dicts)):
            if model_list[m] in model_colors.model_alias:
                model_plot_name = (
                    model_colors.model_alias[model_list[m]]['plot_name']
                )
            else:
                model_plot_name = model_list[m]
            if str(model_list[m]) not in pivot_metric1:
                continue
            y_vals_metric1 = pivot_metric1[str(model_list[m])].values
            y_vals_metric1_mean = np.nanmean(y_vals_metric1)
            if metric2_name is not None:
                y_vals_metric2 = pivot_metric2[str(model_list[m])].values
                y_vals_metric2_mean = np.nanmean(y_vals_metric2)
            if confidence_intervals:
                y_vals_ci_lower1 = pivot_ci_lower1[
                    str(model_list[m])
                ].values
                y_vals_ci_upper1 = pivot_ci_upper1[
                    str(model_list[m])
                ].values
                if metric2_name is not None:
                    y_vals_ci_lower2 = pivot_ci_lower2[
                        str(model_list[m])
                    ].values
                    y_vals_ci_upper2 = pivot_ci_upper2[
                        str(model_list[m])
                    ].values
            if not y_lim_lock:
                if metric2_name is not None:
                    y_vals_both_metrics = np.concatenate((y_vals_metric1, y_vals_metric2))
                    if np.any(y_vals_both_metrics != np.inf):
                        y_vals_metric_min = np.nanmin(y_vals_both_metrics[y_vals_both_metrics != np.inf])
                        y_vals_metric_max = np.nanmax(y_vals_both_metrics[y_vals_both_metrics != np.inf])
                    else:
                        y_vals_metric_min = np.nanmin(y_vals_both_metrics)
                        y_vals_metric_max = np.nanmax(y_vals_both_metrics)
                else:
                    if np.any(y_vals_metric1 != np.inf):
                        y_vals_metric_min = np.nanmin(y_vals_metric1[y_vals_metric1 != np.inf])
                        y_vals_metric_max = np.nanmax(y_vals_metric1[y_vals_metric1 != np.inf])
                    else:
                        y_vals_metric_min = np.nanmin(y_vals_metric1)
                        y_vals_metric_max = np.nanmax(y_vals_metric1)
                if n_mods == 0:
                    y_mod_min = y_vals_metric_min
                    y_mod_max = y_vals_metric_max
                    counts = pivot_counts[str(model_list[m])].values
                    n_mods+=1
                else:
                    if math.isinf(y_mod_min):
                        y_mod_min = y_vals_metric_min
                    else:
                        y_mod_min = np.nanmin([y_mod_min, y_vals_metric_min])
                    if math.isinf(y_mod_max):
                        y_mod_max = y_vals_metric_max
                    else:
                        y_mod_max = np.nanmax([y_mod_max, y_vals_metric_max])
                if (y_vals_metric_min > y_min_limit 
                        and y_vals_metric_min <= y_mod_min):
                    y_min = y_vals_metric_min
                if (y_vals_metric_max < y_max_limit 
                        and y_vals_metric_max >= y_mod_max):
                    y_max = y_vals_metric_max
            if np.abs(y_vals_metric1_mean) < 1E4:
                metric1_mean_fmt_string = f'{y_vals_metric1_mean:.2f}'
            else:
                metric1_mean_fmt_string = f'{y_vals_metric1_mean:.2E}'
            if plot_reference[0]:
                if not plotted_reference[0]:
                    ref_color_dict = model_colors.get_color_dict('obs')
                    plt.plot(
                        x_vals1.tolist(), reference1,
                        marker=ref_color_dict['marker'],
                        c=ref_color_dict['color'], mew=2., mec='white',
                        figure=fig, ms=ref_color_dict['markersize'], ls='solid',
                        lw=ref_color_dict['linewidth']
                    )
                    plotted_reference[0] = True
            else:
                plt.plot(
                    x_vals1.tolist(), y_vals_metric1, 
                    marker=mod_setting_dicts[m]['marker'], 
                    c=mod_setting_dicts[m]['color'], mew=2., mec='white', 
                    figure=fig, ms=mod_setting_dicts[m]['markersize'], ls='solid', 
                    lw=mod_setting_dicts[m]['linewidth']
                )
            if metric2_name is not None:
                if np.abs(y_vals_metric2_mean) < 1E4:
                    metric2_mean_fmt_string = f'{y_vals_metric2_mean:.2f}'
                else:
                    metric2_mean_fmt_string = f'{y_vals_metric2_mean:.2E}'
                if plot_reference[1]:
                    if not plotted_reference[1]:
                        ref_color_dict = model_colors.get_color_dict('obs')
                        plt.plot(
                            x_vals2.tolist(), reference2,
                            marker=ref_color_dict['marker'],
                            c=ref_color_dict['color'], mew=2., mec='white',
                            figure=fig, ms=ref_color_dict['markersize'], ls='dashed',
                            lw=ref_color_dict['linewidth']
                        )
                        plotted_reference[1] = True
                else:
                    plt.plot(
                        x_vals2.tolist(), y_vals_metric2, 
                        marker=mod_setting_dicts[m]['marker'], 
                        c=mod_setting_dicts[m]['color'], mew=2., mec='white', 
                        figure=fig, ms=mod_setting_dicts[m]['markersize'], 
                        ls='dashed', lw=mod_setting_dicts[m]['linewidth']
                    )
            if confidence_intervals:
                if plot_reference[0]:
                    if not plotted_reference_CIs[0]:
                        ref_color_dict = model_colors.get_color_dict('obs')
                        plt.errorbar(
                            x_vals1.tolist(), reference1,
                            yerr=[np.abs(reference_ci_lower1), reference_ci_upper1],
                            fmt='none', ecolor=ref_color_dict['color'],
                            elinewidth=ref_color_dict['linewidth'],
                            capsize=10., capthick=ref_color_dict['linewidth'],
                            alpha=.70, zorder=0
                        )
                        plotted_reference_CIs[0] = True
                else:
                    plt.errorbar(
                        x_vals1.tolist(), y_vals_metric1,
                        yerr=[np.abs(y_vals_ci_lower1), y_vals_ci_upper1],
                        fmt='none', ecolor=mod_setting_dicts[m]['color'],
                        elinewidth=mod_setting_dicts[m]['linewidth'],
                        capsize=10., capthick=mod_setting_dicts[m]['linewidth'],
                        alpha=.70, zorder=0
                    )
                if metric2_name is not None:
                    if plot_reference[1]:
                        if not plotted_reference_CIs[1]:
                            ref_color_dict = model_colors.get_color_dict('obs')
                            plt.errorbar(
                                x_vals2.tolist(), reference2,
                                yerr=[np.abs(reference_ci_lower2), reference_ci_upper2],
                                fmt='none', ecolor=ref_color_dict['color'],
                                elinewidth=ref_color_dict['linewidth'],
                                capsize=10., capthick=ref_color_dict['linewidth'],
                                alpha=.70, zorder=0
                            )
                            plotted_reference_CIs[1] = True
                    else:
                        plt.errorbar(
                            x_vals2.tolist(), y_vals_metric2,
                            yerr=[np.abs(y_vals_ci_lower2), y_vals_ci_upper2],
                            fmt='none', ecolor=mod_setting_dicts[m]['color'],
                            elinewidth=mod_setting_dicts[m]['linewidth'],
                            capsize=10., capthick=mod_setting_dicts[m]['linewidth'],
                            alpha=.70, zorder=0
                        )
            handles+=[
                self.f(
                    mod_setting_dicts[m]['marker'], mod_setting_dicts[m]['color'],
                    'solid', mod_setting_dicts[m]['linewidth'], 
                    mod_setting_dicts[m]['markersize'], 'white'
                )
            ]
            if display_averages:
                if metric2_name is not None:
                    labels+=[
                        f'{model_plot_name} ({metric1_mean_fmt_string},'
                        + f' {metric2_mean_fmt_string})'
                    ]
                else:
                    labels+=[
                        f'{model_plot_name} ({metric1_mean_fmt_string})'
                    ]
            else:
                labels+=[f'{model_plot_name}']

        return (fig, y_min, y_max, handles, labels)
