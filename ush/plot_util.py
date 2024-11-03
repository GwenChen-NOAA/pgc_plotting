#!/usr/bin/env python3
# =============================================================================
#
# NAME: plot_util.py
# CONTRIBUTOR(S): Marcel Caron, marcel.caron@noaa.gov, NOAA/NWS/NCEP/EMC-VPPPGB
# PURPOSE: Plotting tools for Precipitation Grand Challenge (PGC) plotting 
#          scripts
#
# =============================================================================

import os
import sys
from datetime import datetime, timedelta as td
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
"""!@namespace plot_util
   @brief Provides utility functions for PGC plotting use case
"""


def aggregate_stats(df_groups, model_list, date_type, line_type, plot_type, 
                    sample_equalization=True, keep_shared_events_only=True, 
                    delete_intermed_data=False):
    if str(line_type).upper() == 'CTC':
        df_aggregated = df_groups.sum()
    else:
        df_aggregated = df_groups.mean()
    if sample_equalization:
        df_aggregated['COUNTS']=df_groups.size()
    if plot_type == 'timeseries':
        if keep_shared_events_only:
            # Remove data if they exist for some but not all models at some value of 
            # the indep. variable. Otherwise plot_util.calculate_stat will throw an 
            # error
            df_split = [
                df_aggregated.xs(str(model)) for model in model_list
            ]
            df_reduced = reduce(
                lambda x,y: pd.merge(
                    x, y, on=str(date_type).upper(), how='inner'
                ), 
                df_split
            )
        
            df_aggregated = df_aggregated[
                df_aggregated.index.get_level_values(str(date_type).upper())
                .isin(df_reduced.index)
            ]
    elif plot_type == 'fhrmean':
        df_aggregated = df_aggregated.reindex(
            pd.MultiIndex.from_product(
                [
                    np.unique(df_aggregated.index.get_level_values('MODEL')), 
                    np.unique(df_aggregated.index.get_level_values('LEAD_HOURS'))
                ], 
                names=['MODEL','LEAD_HOURS']
            ), 
            fill_value=np.nan
        )
        df_agg_no_nan_rows = ~df_aggregated.isna().any(axis=1)
        max_leads = [
            df_aggregated[df_agg_no_nan_rows].iloc[
                df_aggregated[df_agg_no_nan_rows]
                .index.get_level_values('MODEL') == model
            ].index.get_level_values('LEAD_HOURS').max() for model in model_list
        ]
        remove_rows_by_lead = []
        for m, model in enumerate(model_list):
            df_model_group = df_aggregated.iloc[
                df_aggregated.index.get_level_values('MODEL') == model
            ]
            rows_with_nans = df_model_group.index.get_level_values('LEAD_HOURS')[
                df_model_group.isna().any(axis=1)
            ]
            remove_rows_by_lead_m = [
                int(lead) for lead in rows_with_nans if lead < max_leads[m]
            ]
            remove_rows_by_lead = np.concatenate(
                (remove_rows_by_lead, remove_rows_by_lead_m)
            )
        if delete_intermed_data:
            df_aggregated = df_aggregated.drop(index=np.unique(remove_rows_by_lead), level=1)
    return df_aggregated

def calculate_average(logger, average_method, stat, model_dataframe,
                      model_stat_values):
   """! Calculate average of dataset
      Args:
         logger               - logging file
         average_method       - string of the method to
                                use to calculate the average
         stat                 - string of the statistic the
                                average is being taken for
         model_dataframe      - dataframe of model .stat
                                columns
         model_stat_values    - array of statistic values
      Returns:
         average_array        - array of average value(s)
   """
   average_array = np.empty_like(model_stat_values[:,0])
   if average_method == 'MEAN':
      for l in range(len(model_stat_values[:, 0])):
         average_array[l] = np.ma.mean(model_stat_values[l,:])
   elif average_method == 'MEDIAN':
      for l in range(len(model_stat_values[:,0])):
         logger.info(np.ma.median(model_stat_values[l,:]))
         average_array[l] = np.ma.median(model_stat_values[l,:])
   elif average_method == 'AGGREGATION':
      ndays = model_dataframe.shape[0]
      model_dataframe_aggsum = (
         model_dataframe.groupby('model_plot_name').agg(['sum'])
      )
      model_dataframe_aggsum.columns = (
         calculate_stat(logger, model_dataframe_aggsum/ndays, stat)
      )
      for l in range(len(avg_array[:,0])):
         average_array[l] = avg_array[l]
   else:
      logger.error("FATAL ERROR: Invalid entry for MEAN_METHOD, "
                   +"use MEAN, MEDIAN, or AGGREGATION")
      exit(1)
   return average_array

def calculate_bootstrap_ci(logger, bs_method, model_data, stat, nrepl, level, 
                           bs_min_samp, conversion):
   """! Calculate the upper and lower bound bootstrap statistic from the 
        data from the read in MET .stat file(s)

        Args:
           bs_method         - string of the method to use to
                               calculate the bootstrap confidence intervals
           model_data        - Dataframe containing the model(s)
                               information from the MET .stat
                               files
           stat              - string of the simple statistic
                               name being plotted
           nrepl             - integer of resamples that create the bootstrap
                               distribution
           level             - float confidence level (0.-1.) of the 
                               confidence interval
           bs_min_samp       - minimum number of samples allowed for 
                               confidence intervals to be computed

        Returns:
           stat_values       - Dataframe of the statistic values lower and
                               upper bounds
           status            - integer to provide the parent script with 
                               information about the outcome of the bootstrap 
                               resampling
   """
   status=0
   model_data.reset_index(inplace=True)
   model_data_columns = model_data.columns.values.tolist()
   if model_data_columns == [ 'TOTAL' ]:
      logger.warning("Empty model_data dataframe")
      line_type = 'NULL'
      if (stat == 'fbar_obar' or stat == 'orate_frate'
            or stat == 'baser_frate'):
         stat_values = model_data.loc[:][['TOTAL']]
         stat_values_fbar = model_data.loc[:]['TOTAL']
         stat_values_obar = model_data.loc[:]['TOTAL']
      else:
         stat_values = model_data.loc[:]['TOTAL']
   else:
      if np.any(conversion):
         bool_convert = True
      else:
         bool_convert = False
      if all(elem in model_data_columns for elem in
            ['FBAR', 'OBAR', 'MAE']):
         line_type = 'SL1L2'
         total = model_data.loc[:]['TOTAL']
         fbar = model_data.loc[:]['FBAR']
         obar = model_data.loc[:]['OBAR']
         fobar = model_data.loc[:]['FOBAR']
         ffbar = model_data.loc[:]['FFBAR']
         oobar = model_data.loc[:]['OOBAR']
         if bool_convert:
             coef, const = conversion
             fbar_og = fbar
             obar_og = obar
             fbar = coef*fbar_og+const
             obar = coef*obar_og+const
             fobar = (
                np.power(coef, 2)*fobar 
                + coef*const*fbar_og 
                + coef*const*obar_og
                + np.power(const, 2)
             )
             ffbar = (
                np.power(coef, 2)*ffbar 
                + 2.*coef*const*fbar_og 
                + np.power(const, 2)
             )
             oobar = (
                np.power(coef, 2)*oobar 
                + 2.*coef*const*obar_og
                + np.power(const, 2)
             )
      elif all(elem in model_data_columns for elem in 
            ['FABAR', 'OABAR', 'MAE']):
         line_type = 'SAL1L2'
         total = model_data.loc[:]['TOTAL']
         fabar = model_data.loc[:]['FABAR']
         oabar = model_data.loc[:]['OABAR']
         foabar = model_data.loc[:]['FOABAR']
         ffabar = model_data.loc[:]['FFABAR']
         ooabar = model_data.loc[:]['OOABAR']
         if bool_convert:
             coef, const = conversion
             fabar = coef*fabar
             oabar = coef*oabar
             foabar = (
                np.power(coef, 2)*foabar 
             )
             ffabar = (
                np.power(coef, 2)*ffabar 
             )
             ooabar = (
                np.power(coef, 2)*ooabar 
             )
      elif all(elem in model_data_columns for elem in
            ['UFBAR', 'VFBAR']):
         line_type = 'VL1L2'
         total = model_data.loc[:]['TOTAL']
         ufbar = model_data.loc[:]['UFBAR']
         vfbar = model_data.loc[:]['VFBAR']
         uobar = model_data.loc[:]['UOBAR']
         vobar = model_data.loc[:]['VOBAR']
         uvfobar = model_data.loc[:]['UVFOBAR']
         uvffbar = model_data.loc[:]['UVFFBAR']
         uvoobar = model_data.loc[:]['UVOOBAR']
         if bool_convert:
             coef, const = conversion
             ufbar_og = ufbar
             vfbar_og = vfbar
             uobar_og = uobar
             vobar_og = vobar
             ufbar = coef*ufbar_og+const
             vfbar = coef*vfbar_og+const
             uobar = coef*uobar_og+const
             vobar = coef*vobar_og+const
             uvfobar = (
                np.power(coef, 2)*uvfobar 
                + coef*const*(ufbar_og + uobar_og + vfbar_og + vobar_og) 
                + np.power(const, 2)
             )
             uvffbar = (
                np.power(coef, 2)*uvffbar 
                + 2.*coef*const*(ufbar_og + vfbar_og) 
                + np.power(const, 2)
             )
             uvoobar = (
                np.power(coef, 2)*uvoobar 
                + 2.*coef*const*(uobar_og + vobar_og) 
                + np.power(const, 2)
             )
      elif all(elem in model_data_columns for elem in 
            ['UFABAR', 'VFABAR']):
         line_type = 'VAL1L2'
         total = model_data.loc[:]['TOTAL']
         ufabar = model_data.loc[:]['UFABAR']
         vfabar = model_data.loc[:]['VFABAR']
         uoabar = model_data.loc[:]['UOABAR']
         voabar = model_data.loc[:]['VOABAR']
         uvfoabar = model_data.loc[:]['UVFOABAR']
         uvffabar = model_data.loc[:]['UVFFABAR']
         uvooabar = model_data.loc[:]['UVOOABAR']
         if bool_convert:
             coef, const = conversion
             ufabar = coef*ufabar
             vfabar = coef*vfabar
             uoabar = coef*uoabar
             voabar = coef*voabar
             uvfoabar = (
                np.power(coef, 2)*uvfoabar 
             )
             uvffabar = (
                np.power(coef, 2)*uvffabar 
             )
             uvooabar = (
                np.power(coef, 2)*uvooabar 
             )
      elif all(elem in model_data_columns for elem in
            ['VDIFF_SPEED', 'VDIFF_DIR']):
         line_type = 'VCNT'
         total = model_data.loc[:]['TOTAL']
         fbar = model_data.loc[:]['FBAR']
         obar = model_data.loc[:]['OBAR']
         fs_rms = model_data.loc[:]['FS_RMS']
         os_rms = model_data.loc[:]['OS_RMS']
         msve = model_data.loc[:]['MSVE']
         rmsve = model_data.loc[:]['RMSVE']
         fstdev = model_data.loc[:]['FSTDEV']
         ostdev = model_data.loc[:]['OSTDEV']
         fdir = model_data.loc[:]['FDIR']
         odir = model_data.loc[:]['ODIR']
         fbar_speed = model_data.loc[:]['FBAR_SPEED']
         obar_speed = model_data.loc[:]['OBAR_SPEED']
         vdiff_speed = model_data.loc[:]['VDIFF_SPEED']
         vdiff_dir = model_data.loc[:]['VDIFF_DIR']
         speed_err = model_data.loc[:]['SPEED_ERR']
         dir_err = model_data.loc[:]['DIR_ERR']
         if bool_convert:
            logger.error(
               f"FATAL ERROR: Cannot convert columns for line_type \"{line_type}\""
            )
            exit(1) 
      elif all(elem in model_data_columns for elem in
            ['FY_OY', 'FN_ON']):
         line_type = 'CTC'
         total = model_data.loc[:]['TOTAL']
         fy_oy = model_data.loc[:]['FY_OY']
         fy_on = model_data.loc[:]['FY_ON']
         fn_oy = model_data.loc[:]['FN_OY']
         fn_on = model_data.loc[:]['FN_ON']
      elif all(elem in model_data_columns for elem in 
            ['N_CAT', 'F0_O0']):
         line_type = 'MCTC'
         total = model_data.loc[:]['TOTAL']
         counts = model_data.loc[:]['COUNTS']
         n_cat = model_data.loc[:]['N_CAT']/counts
         i_val = model_data.loc[:]['i_vals']/counts
         fy_oy_cols = get_MCTC_cols_for_sum(n_cat, i_val, 'fy_oy')
         fy_on_cols = get_MCTC_cols_for_sum(n_cat, i_val, 'fy_on')
         fn_oy_cols = get_MCTC_cols_for_sum(n_cat, i_val, 'fn_oy')
         fy_oy = np.array(
            [
                model_data.reset_index().loc[i, fy_oy_cols[i]].sum() 
                for i in model_data.reset_index().index
            ]
         )
         fy_on = np.array(
            [
                model_data.reset_index().loc[i, fy_on_cols[i]].sum() 
                for i in model_data.reset_index.index
            ]
         )
         fn_oy = np.array(
            [
                model_data.reset_index().loc[i, fn_oy_cols[i]].sum() 
                for i in model_data.reset_index().index
            ]
         )
         fn_on = total - fy_oy - fy_on - fn_oy
      elif all(elem in model_data_columns for elem in
            ['FBS','FSS','AFSS','UFSS','F_RATE','O_RATE']):
         line_type = 'NBRCNT'
         total = model_data.loc[:]['TOTAL']
         fbs = model_data.loc[:]['FBS']
         fss = model_data.loc[:]['FSS']
         afss = model_data.loc[:]['AFSS']
         ufss = model_data.loc[:]['UFSS']
         frate = model_data.loc[:]['F_RATE']
         orate = model_data.loc[:]['O_RATE']
      else:
         logger.error("FATAL ERROR: Could not recognize line type from columns")
         exit(1)
   if str(bs_method).upper() == 'MATCHED_PAIRS':
      if total.sum() < bs_min_samp:
         logger.warning(f"Sample too small for bootstrapping. (Matched pairs"
                        + f" sample size: {total.sum()}; minimum sample"
                        + f" size: {bs_min_samp}")
         status = 1
         return pd.DataFrame(
            dict(CI_LOWER=[np.nan], CI_UPPER=[np.nan], STATUS=[status])
         )
      lower_pctile = 100.*((1.-level)/2.)
      upper_pctile = 100.-lower_pctile
      if line_type in ['MCTC','CTC','NBRCTC']:
         fy_oy_all = fy_oy.sum()
         fy_on_all = fy_on.sum()
         fn_oy_all = fn_oy.sum()
         fn_on_all = fn_on.sum()
         total_all = total.sum()
         ctc_all = np.array([fy_oy_all, fy_on_all, fn_oy_all, fn_on_all])
         prob_ctc_all = ctc_all/total_all.astype(float)
         # sample over events in the aggregated contingency table
         fy_oy_samp,fy_on_samp,fn_oy_samp,fn_on_samp = np.random.multinomial(
            total_all, 
            prob_ctc_all, 
            size=nrepl
         ).T
      elif line_type == 'SL1L2':
         fo_matched_est = []
         fvar = ffbar-fbar*fbar
         ovar = oobar-obar*obar
         focovar = fobar-fbar*obar
         for i, _ in enumerate(total):
            fo_matched_est_i = np.random.multivariate_normal(
               [fbar[i], obar[i]], 
               [[fvar[i],focovar[i]],[focovar[i],ovar[i]]], 
               size=int(total[i])
            )
            fo_matched_est.append(fo_matched_est_i)
         fo_matched_est = np.vstack(fo_matched_est)
         fbar_est_mean = fo_matched_est[:,0].mean()
         obar_est_mean = fo_matched_est[:,1].mean()
         fobar_est_mean = np.mean(np.prod(fo_matched_est, axis=1))
         ffbar_est_mean = np.mean(fo_matched_est[:,0]*fo_matched_est[:,0])
         oobar_est_mean = np.mean(fo_matched_est[:,1]*fo_matched_est[:,1])
         max_mem_per_array = 32 # MB
         max_array_size = max_mem_per_array*1E6/8
         batch_size = int(max_array_size/len(fo_matched_est))
         fbar_est_samples = []
         obar_est_samples = []
         fobar_est_samples = []
         ffbar_est_samples = []
         oobar_est_samples = []
         # attempt to bootstrap in batches to save time
         # if sampling array is too large for batches, traditional bootstrap
         if batch_size <= 1:
            for _ in range(nrepl):
               fo_matched_indices = np.random.choice(
                  len(fo_matched_est), 
                  size=fo_matched_est.size, 
                  replace=True
               )
               f_est_bs, o_est_bs = fo_matched_est[fo_matched_indices].T
               fbar_est_samples.append(f_est_bs.mean())
               obar_est_samples.append(o_est_bs.mean())
               fobar_est_samples.append(np.mean(f_est_bs*o_est_bs))
               ffbar_est_samples.append(np.mean(f_est_bs*f_est_bs))
               oobar_est_samples.append(np.mean(o_est_bs*o_est_bs))
            fbar_est_samp = np.array(fbar_est_samples)
            obar_est_samp = np.array(obar_est_samples)
            fobar_est_samp = np.array(fobar_est_samples)
            ffbar_est_samp = np.array(ffbar_est_samples)
            oobar_est_samp = np.array(oobar_est_samples)
         else:
            rep_arr = np.arange(0,nrepl)
            for b in range(0, nrepl, batch_size):
               curr_batch_size = len(rep_arr[b:b+batch_size])
               idxs = [
                  np.random.choice(
                     len(fo_matched_est), 
                     size=fo_matched_est.size, 
                     replace=True
                  ) 
                  for _ in range(curr_batch_size)
               ]
               f_est_bs, o_est_bs = [
                  np.take(fo_matched_est.T[i], idxs) for i in [0,1]
               ]
               fbar_est_samples.append(f_est_bs.mean(axis=1))
               obar_est_samples.append(o_est_bs.mean(axis=1))
               fobar_est_samples.append((f_est_bs*o_est_bs).mean(axis=1))
               ffbar_est_samples.append((f_est_bs*f_est_bs).mean(axis=1))
               oobar_est_samples.append((o_est_bs*o_est_bs).mean(axis=1))
            fbar_est_samp = np.concatenate((fbar_est_samples))
            obar_est_samp = np.concatenate((obar_est_samples))
            fobar_est_samp = np.concatenate((fobar_est_samples))
            ffbar_est_samp = np.concatenate((ffbar_est_samples))
            oobar_est_samp = np.concatenate((oobar_est_samples))
      else:
         logger.error(
            "FATAL ERROR: "
            + line_type
            + f" is not currently a valid option for bootstrapping {bs_method}"
         )
         exit(1)
   elif str(bs_method).upper() == 'FORECASTS':
      if total.size < bs_min_samp:
         logger.warning(f"Sample too small for bootstrapping. (Forecasts"
                        + f" sample size: {total.size}; minimum sample"
                        + f" size: {bs_min_samp}")
         status = 1
         return pd.DataFrame(
            dict(CI_LOWER=[np.nan], CI_UPPER=[np.nan], STATUS=[status])
         )
      lower_pctile = 100.*((1.-level)/2.)
      upper_pctile = 100.-lower_pctile
      if line_type in ['MCTC','CTC','NBRCTC']:
         ctc = np.array([fy_oy, fy_on, fn_oy, fn_on])
         fy_oy_samp, fy_on_samp, fn_oy_samp, fn_on_samp = [
            [] for item in range(4)
         ]
         for _ in range(nrepl):
            ctc_bs = ctc.T[
               np.random.choice(
                  range(len(ctc.T)), 
                  size=len(ctc.T), 
                  replace=True
               )
            ].sum(axis=0)
            fy_oy_samp.append(ctc_bs[0])
            fy_on_samp.append(ctc_bs[1])
            fn_oy_samp.append(ctc_bs[2])
            fn_on_samp.append(ctc_bs[3])
         fy_oy_samp = np.array(fy_oy_samp)
         fy_on_samp = np.array(fy_on_samp)
         fn_oy_samp = np.array(fn_oy_samp)
         fn_on_samp = np.array(fn_on_samp)
      elif line_type == 'SL1L2':
         fbar_est_mean = fbar.mean()
         obar_est_mean = obar.mean()
         fobar_est_mean = fobar.mean()
         ffbar_est_mean = ffbar.mean()
         oobar_est_mean = oobar.mean()
         max_mem_per_array = 32 # MB
         max_array_size = max_mem_per_array*1E6/8
         batch_size = int(max_array_size/len(fbar))
         fbar_samples = []
         obar_samples = []
         fobar_samples = []
         ffbar_samples = []
         oobar_samples = []
         # attempt to bootstrap in batches to save time
         # if sampling array is too large for batches, traditional bootstrap
         if batch_size <= 1:
            for _ in range(nrepl):
               idx = np.random.choice(
                  len(fbar), 
                  size=fbar.size, 
                  replace=True
               )
               fbar_bs, obar_bs, fobar_bs, ffbar_bs, oobar_bs = [
                  summary_stat[idx].T 
                  for summary_stat in [fbar, obar, fobar, ffbar, oobar]
               ]
               fbar_samples.append(fbar_bs.mean())
               obar_samples.append(obar_bs.mean())
               fobar_samples.append(fobar_bs.mean())
               ffbar_samples.append(ffbar_bs.mean())
               oobar_samples.append(oobar_bs.mean())
            fbar_est_samp = np.array(fbar_samples)
            obar_est_samp = np.array(obar_samples)
            fobar_est_samp = np.array(fobar_samples)
            ffbar_est_samp = np.array(ffbar_samples)
            oobar_est_samp = np.array(oobar_samples)
         else:
            rep_arr = np.arange(0,nrepl)
            for b in range(0, nrepl, batch_size):
               curr_batch_size = len(rep_arr[b:b+batch_size])
               idxs = [
                  np.random.choice(
                     len(fbar), 
                     size=fbar.size, 
                     replace=True
                  ) 
                  for _ in range(curr_batch_size)
               ]
               fbar_bs, obar_bs, fobar_bs, ffbar_bs, oobar_bs = [
                  np.take(np.array(summary_stat), idxs) 
                  for s, summary_stat in enumerate([fbar, obar, fobar, ffbar, oobar])
               ]
               fbar_samples.append(fbar_bs.mean(axis=1))
               obar_samples.append(obar_bs.mean(axis=1))
               fobar_samples.append(fobar_bs.mean(axis=1))
               ffbar_samples.append(ffbar_bs.mean(axis=1))
               oobar_samples.append(oobar_bs.mean(axis=1))
            fbar_est_samp = np.concatenate((fbar_samples))
            obar_est_samp = np.concatenate((obar_samples))
            fobar_est_samp = np.concatenate((fobar_samples))
            ffbar_est_samp = np.concatenate((ffbar_samples))
            oobar_est_samp = np.concatenate((oobar_samples))
      elif line_type == 'NBRCNT':
         fbs_est_mean = fbs.mean()
         fss_est_mean = fss.mean()
         afss_est_mean = afss.mean()
         ufss_est_mean = ufss.mean()
         frate_est_mean = frate.mean()
         orate_est_mean = orate.mean()
         max_mem_per_array = 32 # MB
         max_array_size = max_mem_per_array*1E6/8
         batch_size = int(max_array_size/len(fbs))
         fbs_samples = []
         fss_samples = []
         afss_samples = []
         ufss_samples = []
         frate_samples = []
         orate_samples = []
         # attempt to bootstrap in batches to save time
         # if sampling array is too large for batches, traditional bootstrap
         if batch_size <= 1:
            for _ in range(nrepl):
               idx = np.random.choice(
                  len(fbs),
                  size=fbs.size,
                  replace=True
               )
               fbs_bs, fss_bs, afss_bs, ufss_bs, frate_bs, orate_bs = [
                  summary_stat[idx].T
                  for summary_stat in [fbs, fss, afss, ufss, frate, orate]
               ]
               fbs_samples.append(fbs_bs.mean())
               fss_samples.append(fss_bs.mean())
               afss_samples.append(afss_bs.mean())
               ufss_samples.append(ufss_bs.mean())
               frate_samples.append(frate_bs.mean())
               orate_samples.append(orate_bs.mean())
            fbs_est_samp = np.array(fbs_samples)
            fss_est_samp = np.array(fss_samples)
            afss_est_samp = np.array(afss_samples)
            ufss_est_samp = np.array(ufss_samples)
            frate_est_samp = np.array(frate_samples)
            orate_est_samp = np.array(orate_samples)
         else:
            rep_arr = np.arange(0,nrepl)
            for b in range(0, nrepl, batch_size):
               curr_batch_size = len(rep_arr[b:b+batch_size])
               idxs = [
                  np.random.choice(
                     len(fbs),
                     size=fbs.size,
                     replace=True
                  )
                  for _ in range(curr_batch_size)
               ]
               fbs_bs, fss_bs, afss_bs, ufss_bs, frate_bs, orate_bs = [
                  np.take(np.array(summary_stat), idxs)
                  for s, summary_stat in enumerate([fbs, fss, afss, ufss, frate, orate])
               ]
               fbs_samples.append(fbs_bs.mean(axis=1))
               fss_samples.append(fss_bs.mean(axis=1))
               afss_samples.append(afss_bs.mean(axis=1))
               ufss_samples.append(ufss_bs.mean(axis=1))
               frate_samples.append(frate_bs.mean(axis=1))
               orate_samples.append(orate_bs.mean(axis=1))
            fbs_est_samp = np.concatenate((fbs_samples))
            fss_est_samp = np.concatenate((fss_samples))
            afss_est_samp = np.concatenate((afss_samples))
            ufss_est_samp = np.concatenate((ufss_samples))
            frate_est_samp = np.concatenate((frate_samples))
            orate_est_samp = np.concatenate((orate_samples))
      else:
         logger.error("FATAL ERROR: "+line_type+" is not currently a valid option")
         exit(1)
   else:
      logger.error("FATAL ERROR: "+bs_method+" is not a valid option")
      exit(1)
   if stat == 'me':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            stat_values_mean = np.mean(fbar_est_mean) - np.mean(obar_est_mean)
            stat_values = fbar_est_samp - obar_est_samp
         elif line_type in ['MCTC','CTC','NBRCTC']:
            stat_values = (fy_oy_samp + fy_on_samp)/(fy_oy_samp + fn_oy_samp)
   elif stat == 'rmse':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            stat_values_pre_mean = np.sqrt(
               ffbar_est_mean + oobar_est_mean - 2*fobar_est_mean
            )
            stat_values_mean = np.mean(stat_values_pre_mean)
            stat_values = np.sqrt(
               ffbar_est_samp + oobar_est_samp - 2*fobar_est_samp
            )
   elif stat == 'bcrmse':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            var_f_mean = (
               np.mean(ffbar_est_mean) 
               - np.mean(fbar_est_mean)*np.mean(fbar_est_mean)
            )
            var_o_mean = (
               np.mean(oobar_est_mean) 
               - np.mean(obar_est_mean)*np.mean(obar_est_mean)
            )
            covar_mean = (
               np.mean(fobar_est_mean) 
               - np.mean(fbar_est_mean)*np.mean(obar_est_mean)
            )
            stat_values_mean = np.sqrt(var_f_mean+var_o_mean-2*covar_mean)
            var_f = ffbar_est_samp - fbar_est_samp*fbar_est_samp
            var_o = oobar_est_samp - obar_est_samp*obar_est_samp
            covar = fobar_est_samp - fbar_est_samp*obar_est_samp
            stat_values = np.sqrt(var_f+var_o-2*covar)
   elif stat == 'msess':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            mse_mean = ffbar_est_mean + oobar_est_mean - 2*fobar_est_mean
            var_o_mean = oobar_est_mean - obar_est_mean*obar_est_mean
            stat_values_pre_mean = 1 - mse_mean/var_o_mean
            stat_values_mean = np.mean(stat_values_pre_mean)
            mse = ffbar_est_samp + oobar_est_samp - 2*fobar_est_samp
            var_o = oobar_est_samp - obar_est_samp*obar_est_samp
            stat_values = 1 - mse/var_o
   elif stat == 'rsd':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            var_f_mean = ffbar_est_mean - fbar_est_mean*fbar_est_mean
            var_o_mean = oobar_est_mean - obar_est_mean*obar_est_mean
            stat_values_pre_mean = np.sqrt(var_f_mean)/np.sqrt(var_o_mean)
            stat_values_mean = np.mean(stat_values_pre_mean)
            var_f = ffbar_est_samp - fbar_est_samp*fbar_est_samp
            var_o = oobar_est_samp - obar_est_samp*obar_est_samp
            stat_values = np.sqrt(var_f)/np.sqrt(var_o)
   elif stat == 'rmse_md':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            stat_values_pre_mean = np.sqrt((fbar_est_mean-obar_est_mean)**2)
            stat_values_mean = np.mean(stat_values_pre_mean)
            stat_values = np.sqrt((fbar_est_samp-obar_est_samp)**2)
   elif stat == 'rmse_pv':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            var_f_mean = ffbar_est_mean - fbar_est_mean**2
            var_o_mean = oobar_est_mean - obar_est_mean**2
            covar_mean = fobar_est_mean - fbar_est_mean*obar_est_mean
            R_mean = covar_mean/np.sqrt(var_f_mean*var_o_mean)
            stat_values_pre_mean = np.sqrt(
               var_f_mean 
               + var_o_mean 
               - 2*np.sqrt(var_f_mean*var_o_mean)*R_mean
            )
            stat_values_mean = np.mean(stat_values_pre_mean)
            var_f = ffbar_est_samp - fbar_est_samp**2
            var_o = oobar_est_samp - obar_est_samp**2
            covar = fobar_est_samp - fbar_est_samp*obar_est_samp
            R = covar/np.sqrt(var_f*var_o)
            stat_values = np.sqrt(var_f + var_o - 2*np.sqrt(var_f*var_o)*R)
   elif stat == 'pcor':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            var_f_mean = ffbar_est_mean - fbar_est_mean*fbar_est_mean
            var_o_mean = oobar_est_mean - obar_est_mean*obar_est_mean
            covar_mean = fobar_est_mean - fbar_est_mean*obar_est_mean
            stat_values_pre_mean = covar_mean/np.sqrt(var_f_mean*var_o_mean)
            stat_values_mean = np.mean(stat_values_pre_mean)
            var_f = ffbar_est_samp - fbar_est_samp*fbar_est_samp
            var_o = oobar_est_samp - obar_est_samp*obar_est_samp
            covar = fobar_est_samp - fbar_est_samp*obar_est_samp
            stat_values = covar/np.sqrt(var_f*var_o)
   elif stat == 'acc':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SAL1L2':
            var_f_mean = ffabar_est_mean - fabar_est_mean*fabar_est_mean
            var_o_mean = ooabar_est_mean - oabar_est_mean*oabar_est_mean
            covar_mean = foabar_est_mean - fabar_est_mean*oabar_est_mean
            stat_values_pre_mean = covar_mean/np.sqrt(var_f_mean*var_o_mean)
            stat_values_mean = np.mean(stat_values_pre_mean)
            var_f = ffabar_est_samp - fabar_est_samp*fabar_est_samp
            var_o = ooabar_est_samp - oabar_est_samp*oabar_est_samp
            covar = foabar_est_samp - fabar_est_samp*oabar_est_samp
            stat_values = covar_samp/np.sqrt(var_f_samp*var_o_samp)
   elif stat == 'fbar':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            stat_values_pre_mean = fbar_est_mean
            stat_values_mean = np.mean(stat_values_pre_mean)
            stat_values = fbar_est_samp
   elif stat == 'obar':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
         if line_type == 'SL1L2':
            stat_values_pre_mean = obar_est_mean
            stat_values_mean = np.mean(stat_values_pre_mean)
            stat_values = obar_est_samp
   elif stat == 'fss':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
          if line_type == 'NBRCNT':
             stat_values_mean = np.mean(fss_est_mean)
             stat_values = fss_est_samp
   elif stat == 'afss':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
          if line_type == 'NBRCNT':
             stat_values_mean = np.mean(afss_est_mean)
             stat_values = afss_est_samp
   elif stat == 'ufss':
      if str(bs_method).upper() in ['MATCHED_PAIRS','FORECASTS']:
          if line_type == 'NBRCNT':
             stat_values_mean = np.mean(ufss_est_mean)
             stat_values = ufss_est_samp
   elif stat == 'orate' or stat == 'baser':
      if line_type in ['MCTC','CTC','NBRCTC']:
         total_mean = (
            np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)+np.sum(fn_on)
         )
         stat_values_mean = (np.sum(fy_oy)+np.sum(fn_oy))/total_mean
         total = (fy_oy_samp + fy_on_samp + fn_oy_samp + fn_on_samp)
         stat_values = (fy_oy_samp + fn_oy_samp)/total
      elif line_type == 'NBRCNT':
         stat_values_mean = np.mean(orate)
         stat_values = orate_est_samp
   elif stat == 'frate':
      if line_type in ['MCTC','CTC','NBRCTC']:
         total_mean = (
            np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)+np.sum(fn_on)
         )
         stat_values_mean = (np.sum(fy_oy)+np.sum(fy_on))/total_mean
         total = (fy_oy_samp + fy_on_samp + fn_oy_samp + fn_on_samp)
         stat_values = (fy_oy_samp + fy_on_samp)/total
      elif line_type == 'NBRCNT':
         stat_values_mean = np.mean(frate)
         stat_values = frate_est_samp
   elif stat == 'orate_frate' or stat == 'baser_frate':
      if line_type in ['MCTC','CTC','NBRCTC']:
         total_mean = (
            np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)+np.sum(fn_on)
         )
         stat_values_fbar_mean = (
            np.sum(fy_oy)+np.sum(fy_on)
         )/total_mean
         stat_values_obar_mean = (
            np.sum(fy_oy)+np.sum(fn_oy)
         )/total_mean
         stat_values_mean = pd.concat(
            [stat_values_fbar_mean, stat_values_obar_mean]
         )
         total = (fy_oy_samp + fy_on_samp + fn_oy_samp + fn_on_samp)
         stat_values_fbar = (fy_oy_samp + fy_on_samp)/total
         stat_values_obar = (fy_oy_samp + fn_oy_samp)/total
         stat_values = pd.concat(
            [stat_values_fbar, stat_values_obar], axis=1
         )
   elif stat == 'accuracy':
      if line_type in ['MCTC','CTC','NBRCTC']:
         total_mean = (
            np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)+np.sum(fn_on)
         )
         stat_values_mean = (
            np.sum(fy_oy)+np.sum(fn_on)
         )/total_mean
         total = (fy_oy_samp + fy_on_samp + fn_oy_samp + fn_on_samp)
         stat_values = (fy_oy_samp + fn_on_samp)/total
   elif stat == 'fbias':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = (
            (np.sum(fy_oy)+np.sum(fy_on))
            /(np.sum(fy_oy)+np.sum(fn_oy))
         )
         stat_values = (fy_oy_samp + fy_on_samp)/(fy_oy_samp + fn_oy_samp)
   elif stat == 'pod' or stat == 'hrate':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = np.sum(fy_oy)/(np.sum(fy_oy)+np.sum(fn_oy))
         stat_values = fy_oy_samp/(fy_oy_samp + fn_oy_samp)
   elif stat == 'pofd' or stat == 'farate':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = np.sum(fy_on)/(np.sum(fy_on)+np.sum(fn_on))
         stat_values = fy_on_samp/(fy_on_samp + fn_on_samp)
   elif stat == 'podn':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = np.sum(fn_on)/(np.sum(fy_on)+np.sum(fn_on))
         stat_values = fn_on_samp/(fy_on_samp + fn_on_samp)
   elif stat == 'faratio':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = np.sum(fy_on)/(np.sum(fy_on)+np.sum(fy_oy))
         stat_values = fy_on_samp/(fy_on_samp + fy_oy_samp)
   elif stat == 'sratio':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = (
            1. - (np.sum(fy_on)/(np.sum(fy_on)+np.sum(fy_oy)))
         )
         stat_values = 1. - (fy_on_samp/(fy_on_samp + fy_oy_samp))
   elif stat == 'csi' or stat == 'ts':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = (
            np.sum(fy_oy)
            /(np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy))
         )
         stat_values = fy_oy_samp/(fy_oy_samp + fy_on_samp + fn_oy_samp)
   elif stat == 'gss' or stat == 'ets':
      if line_type in ['MCTC','CTC','NBRCTC']:
         total_mean = (
            np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)+np.sum(fn_on)
         )
         C_mean = (
            (np.sum(fy_oy)+np.sum(fy_on))
            *(np.sum(fy_oy)+np.sum(fn_oy))
         )/total_mean
         stat_values_mean = (
            (np.sum(fy_oy)-C_mean)
            /(np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)-C_mean)
         )
         total = (fy_oy_samp + fy_on_samp + fn_oy_samp + fn_on_samp)
         C = ((fy_oy_samp + fy_on_samp)*(fy_oy_samp + fn_oy_samp))/total
         stat_values = (
            (fy_oy_samp - C)/(fy_oy_samp + fy_on_samp + fn_oy_samp - C)
         )
   elif stat == 'hk' or stat == 'tss' or stat == 'pss':
      if line_type in ['MCTC','CTC','NBRCTC']:
         stat_values_mean = (
            (np.sum(fy_oy)*np.sum(fn_on)-np.sum(fy_on)*np.sum(fn_oy))
            /(
               (np.sum(fy_oy)+np.sum(fn_oy))
               *(np.sum(fy_on)+np.sum(fn_on))
            )
         )
         stat_values = (
            ((fy_oy_samp*fn_on_samp)-(fy_on_samp*fn_oy_samp))
            /((fy_oy_samp+fn_oy_samp)*(fy_on_samp+fn_on_samp))
         )
   elif stat == 'hss':
      if line_type in ['MCTC','CTC','NBRCTC']:
         total_mean = (
            np.sum(fy_oy)+np.sum(fy_on)+np.sum(fn_oy)+np.sum(fn_on)
         )
         Ca_mean = (
            (np.sum(fy_oy)+np.sum(fy_on))
            *(np.sum(fy_oy)+np.sum(fn_oy))
         )
         Cb_mean = (
            (np.sum(fn_oy)+np.sum(fn_on))
            *(np.sum(fy_on)+np.sum(fn_on))
         )
         C_mean = (Ca_mean + Cb_mean)/total_mean
         stat_values_mean = (
            (np.sum(fy_oy)+np.sum(fn_on)-C_mean)
            /(total_mean-C_mean)
         )
         total = (fy_oy_samp + fy_on_samp + fn_oy_samp + fn_on_samp)
         Ca = (fy_oy_samp+fy_on_samp)*(fy_oy_samp+fn_oy_samp)
         Cb = (fn_oy_samp+fn_on_samp)*(fy_on_samp+fn_on_samp)
         C = (Ca + Cb)/total
         stat_values = (fy_oy_samp + fn_on_samp - C)/(total - C)
   else:
      logger.error("FATAL ERROR: "+stat+" is not a valid option")
      exit(1)
   stat_deltas = stat_values-stat_values_mean
   stat_ci_lower = np.nanpercentile(stat_deltas, lower_pctile)
   stat_ci_upper = np.nanpercentile(stat_deltas, upper_pctile)
   return pd.DataFrame(
      dict(CI_LOWER=[stat_ci_lower], CI_UPPER=[stat_ci_upper], STATUS=[status])
   )

def calculate_stat(logger, model_data, stat, conversion):
   """! Calculate the statistic from the data from the
        read in MET .stat file(s)

        Args:
           model_data        - Dataframe containing the model(s)
                               information from the MET .stat
                               files
           stat              - string of the simple statistic
                               name being plotted

        Returns:
           stat_values       - Dataframe of the statistic values
           stat_plot_name    - string of the formal statistic
                               name being plotted
   """
   model_data_columns = model_data.columns.values.tolist()
   if model_data_columns == [ 'TOTAL' ]:
      logger.warning("Empty model_data dataframe")
      line_type = 'NULL'
      if (stat == 'fbar_obar' or stat == 'orate_frate'
            or stat == 'baser_frate'):
         stat_values = model_data.loc[:][['TOTAL']]
         stat_values_fbar = model_data.loc[:]['TOTAL']
         stat_values_obar = model_data.loc[:]['TOTAL']
      else:
         stat_values = model_data.loc[:]['TOTAL']
   else:
      if np.any(conversion):
         bool_convert = True
      else:
         bool_convert = False
      if all(elem in model_data_columns for elem in
            ['FBAR', 'OBAR', 'MAE']):
         line_type = 'SL1L2'
         fbar = model_data.loc[:]['FBAR']
         obar = model_data.loc[:]['OBAR']
         fobar = model_data.loc[:]['FOBAR']
         ffbar = model_data.loc[:]['FFBAR']
         oobar = model_data.loc[:]['OOBAR']
         if bool_convert:
             coef, const = conversion
             fbar_og = fbar
             obar_og = obar
             fbar = coef*fbar_og+const
             obar = coef*obar_og+const
             fobar = (
                np.power(coef, 2)*fobar 
                + coef*const*fbar_og 
                + coef*const*obar_og
                + np.power(const, 2)
             )
             ffbar = (
                np.power(coef, 2)*ffbar 
                + 2.*coef*const*fbar_og 
                + np.power(const, 2)
             )
             oobar = (
                np.power(coef, 2)*oobar 
                + 2.*coef*const*obar_og
                + np.power(const, 2)
             )
      elif all(elem in model_data_columns for elem in 
            ['FABAR', 'OABAR', 'MAE']):
         line_type = 'SAL1L2'
         fabar = model_data.loc[:]['FABAR']
         oabar = model_data.loc[:]['OABAR']
         foabar = model_data.loc[:]['FOABAR']
         ffabar = model_data.loc[:]['FFABAR']
         ooabar = model_data.loc[:]['OOABAR']
         if bool_convert:
             coef, const = conversion
             fabar = coef*fabar
             oabar = coef*oabar
             foabar = (
                np.power(coef, 2)*foabar 
             )
             ffabar = (
                np.power(coef, 2)*ffabar 
             )
             ooabar = (
                np.power(coef, 2)*ooabar 
             )
      elif all(elem in model_data_columns for elem in
            ['UFBAR', 'VFBAR']):
         line_type = 'VL1L2'
         ufbar = model_data.loc[:]['UFBAR']
         vfbar = model_data.loc[:]['VFBAR']
         uobar = model_data.loc[:]['UOBAR']
         vobar = model_data.loc[:]['VOBAR']
         uvfobar = model_data.loc[:]['UVFOBAR']
         uvffbar = model_data.loc[:]['UVFFBAR']
         uvoobar = model_data.loc[:]['UVOOBAR']
         if bool_convert:
             coef, const = conversion
             ufbar_og = ufbar
             vfbar_og = vfbar
             uobar_og = uobar
             vobar_og = vobar
             ufbar = coef*ufbar_og+const
             vfbar = coef*vfbar_og+const
             uobar = coef*uobar_og+const
             vobar = coef*vobar_og+const
             uvfobar = (
                np.power(coef, 2)*uvfobar 
                + coef*const*(ufbar_og + uobar_og + vfbar_og + vobar_og) 
                + np.power(const, 2)
             )
             uvffbar = (
                np.power(coef, 2)*uvffbar 
                + 2.*coef*const*(ufbar_og + vfbar_og) 
                + np.power(const, 2)
             )
             uvoobar = (
                np.power(coef, 2)*uvoobar 
                + 2.*coef*const*(uobar_og + vobar_og) 
                + np.power(const, 2)
             )
      elif all(elem in model_data_columns for elem in 
            ['UFABAR', 'VFABAR']):
         line_type = 'VAL1L2'
         ufabar = model_data.loc[:]['UFABAR']
         vfabar = model_data.loc[:]['VFABAR']
         uoabar = model_data.loc[:]['UOABAR']
         voabar = model_data.loc[:]['VOABAR']
         uvfoabar = model_data.loc[:]['UVFOABAR']
         uvffabar = model_data.loc[:]['UVFFABAR']
         uvooabar = model_data.loc[:]['UVOOABAR']
         if bool_convert:
             coef, const = conversion
             ufabar = coef*ufabar
             vfabar = coef*vfabar
             uoabar = coef*uoabar
             voabar = coef*voabar
             uvfoabar = (
                np.power(coef, 2)*uvfoabar 
             )
             uvffabar = (
                np.power(coef, 2)*uvffabar 
             )
             uvooabar = (
                np.power(coef, 2)*uvooabar 
             )
      elif all(elem in model_data_columns for elem in
            ['VDIFF_SPEED', 'VDIFF_DIR']):
         line_type = 'VCNT'
         fbar = model_data.loc[:]['FBAR']
         obar = model_data.loc[:]['OBAR']
         fs_rms = model_data.loc[:]['FS_RMS']
         os_rms = model_data.loc[:]['OS_RMS']
         msve = model_data.loc[:]['MSVE']
         rmsve = model_data.loc[:]['RMSVE']
         fstdev = model_data.loc[:]['FSTDEV']
         ostdev = model_data.loc[:]['OSTDEV']
         fdir = model_data.loc[:]['FDIR']
         odir = model_data.loc[:]['ODIR']
         fbar_speed = model_data.loc[:]['FBAR_SPEED']
         obar_speed = model_data.loc[:]['OBAR_SPEED']
         vdiff_speed = model_data.loc[:]['VDIFF_SPEED']
         vdiff_dir = model_data.loc[:]['VDIFF_DIR']
         speed_err = model_data.loc[:]['SPEED_ERR']
         dir_err = model_data.loc[:]['DIR_ERR']
         if bool_convert:
            logger.error(
               f"FATAL ERROR: Cannot convert column units for line_type \"{line_type}\""
            )
            exit(1) 
      elif all(elem in model_data_columns for elem in
            ['FY_OY', 'FN_ON']):
         line_type = 'CTC'
         total = model_data.loc[:]['TOTAL']
         fy_oy = model_data.loc[:]['FY_OY']
         fy_on = model_data.loc[:]['FY_ON']
         fn_oy = model_data.loc[:]['FN_OY']
         fn_on = model_data.loc[:]['FN_ON']
      elif all(elem in model_data_columns for elem in 
            ['N_CAT', 'F0_O0']):
         line_type = 'MCTC'
         total = model_data.loc[:]['TOTAL']
         counts = model_data.loc[:]['COUNTS']
         n_cat = model_data.loc[:]['N_CAT']/counts
         i_val = model_data.loc[:]['i_vals']/counts
         fy_oy_cols = get_MCTC_cols_for_sum(n_cat, i_val, 'fy_oy')
         fy_on_cols = get_MCTC_cols_for_sum(n_cat, i_val, 'fy_on')
         fn_oy_cols = get_MCTC_cols_for_sum(n_cat, i_val, 'fn_oy')
         fy_oy = np.array(
            [
                model_data.reset_index().loc[i, fy_oy_cols[i]].sum() 
                for i in model_data.reset_index().index
            ]
         )
         fy_on = np.array(
            [
                model_data.reset_index().loc[i, fy_on_cols[i]].sum() 
                for i in model_data.reset_index().index
            ]
         )
         fn_oy = np.array(
            [
                model_data.reset_index().loc[i, fn_oy_cols[i]].sum() 
                for i in model_data.reset_index().index
            ]
         )
         fy_oy = pd.DataFrame(fy_oy, index=total.index)[0]
         fy_on = pd.DataFrame(fy_on, index=total.index)[0]
         fn_oy = pd.DataFrame(fn_oy, index=total.index)[0]
         fn_on = total - fy_oy - fy_on - fn_oy
      elif all(elem in model_data_columns for elem in 
            ['FBS','FSS','AFSS','UFSS','F_RATE','O_RATE']):
          line_type = 'NBRCNT'
          total = model_data.loc[:]['TOTAL']
          fbs = model_data.loc[:]['FBS']
          fss = model_data.loc[:]['FSS']
          afss = model_data.loc[:]['AFSS']
          ufss = model_data.loc[:]['UFSS']
          frate = model_data.loc[:]['F_RATE']
          orate = model_data.loc[:]['O_RATE']
      elif all(elem in model_data_columns for elem in
            ['CRPS', 'CRPSS', 'RMSE', 'SPREAD', 'ME', 'MAE']):
         line_type = 'ECNT'
         total  = model_data.loc[:]['TOTAL']
         crps   = model_data.loc[:]['CRPS']
         crpss  = model_data.loc[:]['CRPSS']
         rmse   = model_data.loc[:]['RMSE']
         spread = model_data.loc[:]['SPREAD']
         me     = model_data.loc[:]['ME']
         mae     = model_data.loc[:]['MAE']
      elif all(elem in model_data_columns for elem in
            ['ROC_AUC', 'BRIER', 'BSS', 'BSS_SMPL']):
         line_type = 'PSTD'
         total  = model_data.loc[:]['TOTAL']
         roc_area =  model_data.loc[:]['ROC_AUC']
         bs =  model_data.loc[:]['BRIER']
         bss =  model_data.loc[:]['BSS']
         bss_smpl =  model_data.loc[:]['BSS_SMPL']
      else:
         logger.error("FATAL ERROR: Could not recognize line type from columns")
         exit(1)
   stat_plot_name = get_stat_plot_name(logger, stat)
   if stat == 'me':
      if line_type == 'SL1L2':
         stat_values = fbar - obar
      elif line_type == 'VL1L2':
         stat_values = np.sqrt(uvffbar) - np.sqrt(uvoobar)
      elif line_type == 'VCNT':
         stat_values = fbar - obar
      elif line_type in ['MCTC', 'CTC']:
         stat_values = (fy_oy + fy_on)/(fy_oy + fn_oy)
   elif stat == 'rmse':
      if line_type == 'SL1L2':
         stat_values = np.sqrt(ffbar + oobar - 2*fobar)
      elif line_type == 'VL1L2':
         stat_values = np.sqrt(uvffbar + uvoobar - 2*uvfobar)
      elif line_type == 'ECNT':
         stat_values = rmse
   elif stat == 'crps':
      if line_type == 'ECNT':
        stat_values = crps
   elif stat == 'crpss':
      if line_type == 'ECNT':
        stat_values = crpss
   elif stat == 'spread':
      if line_type == 'ECNT':
        stat_values = spread
   elif stat == 'me':
      if line_type == 'ECNT':
        stat_values = me
   elif stat == 'mae':
      if line_type == 'SL1L2':
        stat_values = mae
      elif line_type == 'ECNT':
        stat_values = mae
   elif stat == 'bs':
      if line_type == 'PSTD':
        stat_values = bs
   elif stat == 'bss':
      if line_type == 'PSTD':
        stat_values = bss
   elif stat == 'bss_smpl':
      if line_type == 'PSTD':
        stat_values = bss_smpl
   elif stat == 'roc_area':
      if line_type == 'PSTD':
        stat_values = roc_area
   elif stat == 'bcrmse':
      if line_type == 'SL1L2':
         var_f = ffbar - fbar*fbar
         var_o = oobar - obar*obar
         covar = fobar - fbar*obar
         stat_values = np.sqrt(var_f + var_o - 2*covar)
      elif line_type == 'VL1L2':
         var_f = uvffbar - ufbar*ufbar - vfbar*vfbar
         var_o = uvoobar - uobar*uobar - vobar*vobar
         covar = uvfobar - ufbar*uobar - vfbar*vobar
         stat_values = np.sqrt(var_f + var_o - 2*covar)
   elif stat == 'msess':
      if line_type == 'SL1L2':
         mse = ffbar + oobar - 2*fobar
         var_o = oobar - obar*obar
         stat_values = 1 - mse/var_o
      elif line_type == 'VL1L2':
         mse = uvffbar + uvoobar - 2*uvfobar
         var_o = uvoobar - uobar*uobar - vobar*vobar
         stat_values = 1 - mse/var_o
   elif stat == 'rsd':
      if line_type == 'SL1L2':
         var_f = ffbar - fbar*fbar
         var_o = oobar - obar*obar
         stat_values = np.sqrt(var_f)/np.sqrt(var_o)
      elif line_type == 'VL1L2':
         var_f = uvffbar - ufbar*ufbar - vfbar*vfbar
         var_o = uvoobar - uobar*uobar - vobar*vobar
         stat_values = np.sqrt(var_f)/np.sqrt(var_o)
      elif line_type == 'VCNT':
         stat_values = fstdev/ostdev
   elif stat == 'rmse_md':
      if line_type == 'SL1L2':
         stat_values = np.sqrt((fbar-obar)**2)
      elif line_type == 'VL1L2':
         stat_values = np.sqrt((ufbar - uobar)**2 + (vfbar - vobar)**2)
   elif stat == 'rmse_pv':
      if line_type == 'SL1L2':
         var_f = ffbar - fbar**2
         var_o = oobar - obar**2
         covar = fobar - fbar*obar
         R = covar/np.sqrt(var_f*var_o)
         stat_values = np.sqrt(var_f + var_o - 2*np.sqrt(var_f*var_o)*R)
      elif line_type == 'VL1L2':
         var_f = uvffbar - ufbar*ufbar - vfbar*vfbar
         var_o = uvoobar - uobar*uobar - vobar*vobar
         covar = uvfobar - ufbar*uobar - vfbar*vobar
         R = covar/np.sqrt(var_f*var_o)
         stat_values = np.sqrt(var_f + var_o - 2*np.sqrt(var_f*var_o)*R)
   elif stat == 'pcor':
      if line_type == 'SL1L2':
         var_f = ffbar - fbar*fbar
         var_o = oobar - obar*obar
         covar = fobar - fbar*obar
         stat_values = covar/np.sqrt(var_f*var_o)
      elif line_type == 'VL1L2':
         var_f = uvffbar - ufbar*ufbar - vfbar*vfbar
         var_o = uvoobar - uobar*uobar - vobar*vobar
         covar = uvfobar - ufbar*uobar - vfbar*vobar
         stat_values = covar/np.sqrt(var_f*var_o)
   elif stat == 'acc':
      if line_type == 'SAL1L2':
         var_f = ffabar - fabar*fabar
         var_o = ooabar - oabar*oabar
         covar = foabar - fabar*oabar
         stat_values = covar/np.sqrt(var_f*var_o)
      if line_type == 'VAL1L2':
         stat_values = uvfoabar/np.sqrt(uvffabar*uvooabar)
   elif stat == 'fbar':
      if line_type == 'SL1L2':
         stat_values = fbar
      elif line_type == 'VL1L2':
         stat_values = np.sqrt(uvffbar)
      elif line_type == 'VCNT':
         stat_values = fbar
   elif stat == 'obar':
      if line_type == 'SL1L2':
         stat_values = obar
      elif line_type == 'VL1L2':
         stat_values = np.sqrt(uvoobar)
      elif line_type == 'VCNT':
         stat_values = obar
   elif stat == 'fbar_obar':
      if line_type == 'SL1L2':
         stat_values = model_data.loc[:][['FBAR', 'OBAR']]
         stat_values_fbar = model_data.loc[:]['FBAR']
         stat_values_obar = model_data.loc[:]['OBAR']
      elif line_type == 'VL1L2':
         stat_values = model_data.loc[:][['UVFFBAR', 'UVOOBAR']]
         stat_values_fbar = np.sqrt(model_data.loc[:]['UVFFBAR'])
         stat_values_obar = np.sqrt(model_data.loc[:]['UVOOBAR'])
      elif line_type == 'VCNT':
         stat_values = model_data.loc[:][['FBAR', 'OBAR']]
         stat_values_fbar = model_data.loc[:]['FBAR']
         stat_values_obar = model_data.loc[:]['OBAR']
   elif stat == 'speed_err':
      if line_type == 'VCNT':
         stat_values = speed_err
   elif stat == 'dir_err':
      if line_type == 'VCNT':
         stat_values = dir_err
   elif stat == 'rmsve':
      if line_type == 'VCNT':
         stat_values = rmsve
   elif stat == 'vdiff_speed':
      if line_type == 'VCNT':
         stat_values = vdiff_speed
   elif stat == 'vdiff_dir':
      if line_type == 'VCNT':
         stat_values = vdiff_dir
   elif stat == 'fbar_obar_speed':
      if line_type == 'VCNT':
         stat_values = model_data.loc[:][('FBAR_SPEED', 'OBAR_SPEED')]
   elif stat == 'fbar_obar_dir':
      if line_type == 'VCNT':
         stat_values = model_data.loc[:][('FDIR', 'ODIR')]
   elif stat == 'fbar_speed':
      if line_type == 'VCNT':
         stat_values = fbar_speed
   elif stat == 'fbar_dir':
      if line_type == 'VCNT':
         stat_values = fdir
   elif stat == 'orate' or stat == 'baser':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = (fy_oy + fn_oy)/total
      elif line_type == 'NBRCNT':
         stat_values = orate
   elif stat == 'frate':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = (fy_oy + fy_on)/total
      elif line_type == 'NBRCNT':
         stat_values = frate
   elif stat == 'fss':
      if line_type == 'NBRCNT':
         stat_values = fss
   elif stat == 'afss':
      if line_type == 'NBRCNT':
         stat_values = afss
   elif stat == 'ufss':
      if line_type == 'NBRCNT':
         stat_values = ufss
   elif stat == 'orate_frate' or stat == 'baser_frate':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values_fbar = (fy_oy + fy_on)/total
         stat_values_obar = (fy_oy + fn_oy)/total
         stat_values = pd.concat(
            [stat_values_fbar, stat_values_obar], axis=1
         )
   elif stat == 'accuracy':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = (fy_oy + fn_on)/total
   elif stat == 'fbias':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = (fy_oy + fy_on)/(fy_oy + fn_oy)
   elif stat == 'pod' or stat == 'hrate':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = fy_oy/(fy_oy + fn_oy)
   elif stat == 'pofd' or stat == 'farate':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = fy_on/(fy_on + fn_on)
   elif stat == 'podn':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = fn_on/(fy_on + fn_on)
   elif stat == 'faratio':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = fy_on/(fy_on + fy_oy)
   elif stat == 'sratio':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = 1. - (fy_on/(fy_on + fy_oy))
   elif stat == 'csi' or stat == 'ts':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = fy_oy/(fy_oy + fy_on + fn_oy)
   elif stat == 'gss' or stat == 'ets':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         C = ((fy_oy + fy_on)*(fy_oy + fn_oy))/total
         stat_values = (fy_oy - C)/(fy_oy + fy_on + fn_oy - C)
   elif stat == 'hk' or stat == 'tss' or stat == 'pss':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         stat_values = (
            ((fy_oy*fn_on)-(fy_on*fn_oy))/((fy_oy+fn_oy)*(fy_on+fn_on))
         )
   elif stat == 'hss':
      if line_type in ['MCTC', 'CTC', 'NBRCTC']:
         Ca = (fy_oy+fy_on)*(fy_oy+fn_oy)
         Cb = (fn_oy+fn_on)*(fy_on+fn_on)
         C = (Ca + Cb)/total
         stat_values = (fy_oy + fn_on - C)/(total - C)
   else:
      logger.error("FATAL ERROR: "+stat+" is not a valid option")
      exit(1)
   nindex = stat_values.index.nlevels
   return stat_values, None, stat_plot_name

def configure_dates_axis(xvals, incr):
    if aggregate_dates_by:
        if aggregate_dates_by in ['m','month']: 
            xticks = x_vals1 
            xtick_labels = [
                datetime.strptime(xtick, '%Y%m').strftime('%b %Y') 
                for xtick in xticks
            ]
        elif aggregate_dates_by in ['Y','year']:
            xticks = x_vals1 
            xtick_labels = xticks
    else:
        xticks = [
            x_val 
            for x_val in daterange(x_vals1[0], x_vals1[-1], td(hours=incr))
        ] 
        xtick_labels = [xtick.strftime('%HZ %m/%d') for xtick in xticks]
    number_of_ticks_dig = np.arange(9, 225, 9, dtype=int)
    show_xtick_every = np.ceil((
        np.digitize(len(xtick_labels), number_of_ticks_dig) + 2
    )/2.)*2
    xtick_labels_with_blanks = ['' for item in xtick_labels]
    for i, item in enumerate(xtick_labels[::int(show_xtick_every)]):
         xtick_labels_with_blanks[int(show_xtick_every)*i] = item
    return xticks, xtick_labels_with_blanks

def configure_stats_axis(y_min, y_max, y_min_limit, y_max_limit, thresh_labels,
                         thresh, metric1_name, metric2_name, metric_long_names, 
                         metrics_using_var_units, units, unit_convert, 
                         reference, var_long_name_key, variable_translator):
    
    # Get axis ticks
    y_range_categories = np.array([
        [np.power(10.,y), 2.*np.power(10.,y)] 
        for y in [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    ]).flatten()
    round_to_nearest_categories = y_range_categories/20.
    if math.isinf(y_min):
        y_min = y_min_limit
    if math.isinf(y_max):
        y_max = y_max_limit
    y_range = y_max-y_min
    round_to_nearest =  round_to_nearest_categories[
        np.digitize(y_range, y_range_categories[:-1])
    ]
    ylim_min = np.floor(y_min/round_to_nearest)*round_to_nearest
    ylim_max = np.ceil(y_max/round_to_nearest)*round_to_nearest
    if len(str(ylim_min)) > 5 and np.abs(ylim_min) < 1.:
        ylim_min = float(
            np.format_float_scientific(ylim_min, unique=False, precision=3)
        )
    if round_to_nearest < 1.:
        y_precision_scale = 100/round_to_nearest
    else:
        y_precision_scale = 1.
    yticks = [
        y_val for y_val 
        in np.arange(
            ylim_min*y_precision_scale, 
            ylim_max*y_precision_scale+round_to_nearest*y_precision_scale, 
            round_to_nearest*y_precision_scale
        )
    ]
    yticks=np.divide(yticks,y_precision_scale)
    ytick_labels = [f'{ytick}' for ytick in yticks]
    show_ytick_every = len(yticks)//10+1
    
    # Generate labels
    ytick_labels_with_blanks = ['' for item in ytick_labels]
    for i, item in enumerate(ytick_labels[::int(show_ytick_every)]):
        ytick_labels_with_blanks[int(show_ytick_every)*i] = item
    if str(var_long_name_key).upper() == 'HGT':
        if str(df['OBS_VAR'].tolist()[0]).upper() in ['CEILING']:
            var_long_name_key = 'HGTCLDCEIL'
        elif str(df['OBS_VAR'].tolist()[0]).upper() in ['HPBL']:
            var_long_name_key = 'HPBL'
    var_long_name = variable_translator[var_long_name_key]
    if unit_convert:
        if thresh and '' not in thresh:
            thresh_labels = [float(tlab) for tlab in thresh_labels]
            thresh_labels = reference.unit_conversions[units]['formula'](
                thresh_labels,
                rounding=True
            )
            thresh_labels = [str(tlab) for tlab in thresh_labels]
        units = reference.unit_conversions[units]['convert_to']
    if units == '-':
        units = ''
    if metric2_name is not None:
        metric1_string, metric2_string = metric_long_names
        if (str(metric1_name).upper() in metrics_using_var_units
                and str(metric2_name).upper() in metrics_using_var_units):
            if units:
                ylabel = f'{var_long_name} ({units})'
            else:
                ylabel = f'{var_long_name} (unitless)'
        else:
            ylabel = f'{metric1_string} and {metric2_string}'
    else:
        metric1_string = metric_long_names[0]
        metric2_string = None
        if str(metric1_name).upper() in metrics_using_var_units:
            if units:
                ylabel = f'{var_long_name} ({units})'
            else:
                ylabel = f'{var_long_name} (unitless)'
        else:
            ylabel = f'{metric1_string}'
    return (
        ylim_min, ylim_max, yticks, ytick_labels_with_blanks, ylabel,
        thresh_labels, metric1_string, metric2_string, units, 
        var_long_name_key, var_long_name
    )

def daterange(start: datetime, end: datetime, td: td) -> datetime:
    curr = start
    while curr <= end:
        yield curr
        curr+=td

def equalize_samples(logger, df, group_by):
    # columns that will be used to drop duplicate rows across model groups
    cols_to_check = [
        key for key in [
            'LEAD_HOURS', 'VALID', 'INIT', 'FCST_THRESH_SYMBOL', 
            'FCST_THRESH_VALUE', 'OBS_LEV']
        if key in df.keys()
    ]
    df_groups = df.groupby(group_by)
    indexes = []
    # List all of the independent variables that are found in the data
    unique_indep_vars = np.unique(np.array(list(df_groups.groups.keys())).T[1])
    for unique_indep_var in unique_indep_vars:
        # Get all groups, in the form of DataFrames, that include the given 
        # independent variable
        dfs = [
            df_groups.get_group(name)[cols_to_check]
            for name in list(df_groups.groups.keys()) 
            if str(name[1]) == str(unique_indep_var)
        ]
        # merge all of these DataFrames together according to 
        # the columns in cols_to_check
        for i, dfs_i in enumerate(dfs):
            if i == 0:
                df_merged = dfs_i
            else:
                df_merged = df_merged.merge(
                    dfs_i, how='inner', indicator=False
                )
            # Reduce the size of the merged df as we go by removing duplicates
            df_merged = df_merged.drop_duplicates()
        # make sure to remove duplicate rows (looking only at the columns in 
        # cols_to_check) to reduce comp time in the next in the next step
        match_these = df_merged.drop_duplicates()
        # Get all the indices for rows in each group that match the merged df
        for dfs_i in dfs:
            for idx, row in dfs_i.iterrows():
                if (
                        row.to_numpy()[1:].tolist() 
                        in match_these.to_numpy()[:,1:].tolist()):
                    indexes.append(idx)
    # Select the matched rows by index among the rows in the original DataFrame
    df_equalized = df.loc[indexes]
    # Remove duplicates again, this time among both the columns 
    # in cols_to_check and the 'MODEL' column, which avoids, say, models with
    # repeated data from multiple entities
    df_equalized = df_equalized.loc[
        df_equalized[cols_to_check+['MODEL']].drop_duplicates().index
    ]
    # Remove duplicates again, this time among both the columns 
    # Regroup the data and move forward with these groups!
    df_equalized_groups = df_equalized.groupby(group_by)
    # Check that groups are indeed equally sized for each independent variable
    df_groups_sizes = df_equalized_groups.size()
    if df_groups_sizes.size > 0:
        df_groups_sizes.index = df_groups_sizes.index.set_levels(
            df_groups_sizes.index.levels[-1].astype(str), level=-1
        )
        data_are_equalized = np.all([
            np.unique(df_groups_sizes.xs(str(unique_indep_var), level=1)).size == 1
            for unique_indep_var 
            in np.unique(np.array(list(df_groups_sizes.keys())).T[1])
        ])
    else:
        logger.info(
            "Sample equalization was successful but resulted in an empty"
            + f" dataframe."
        )
        data_are_equalized = True
    if data_are_equalized:
        logger.info(
            "Data were successfully equalized along the independent"
            + " variable."
        )
        return df_equalized, data_are_equalized
    else:
        logger.warning(
            "FATAL ERROR: Data equalization along the independent variable failed."
        )
        logger.warning(
            "This may be a bug in the verif_plotting code. Please contact"
            + " the verif_plotting code manager about your issue."
        )
        logger.warning(
            "Skipping equalization.  Sample sizes will not be plotted."
        )
        return df, data_are_equalized

def filter_by_lead(logger, df, flead):
    if isinstance(flead, list):
        if len(flead) <= 3:
            if len(flead) > 1:
                frange_phrase = 's '+', '.join([str(f) for f in flead])
            else:
                frange_phrase = ' '+', '.join([str(f) for f in flead])
            frange_save_phrase = '-'.join([str(f).zfill(3) for f in flead])
        else:
            frange_phrase = f's {flead[0]}'+u'\u2013'+f'{flead[-1]}'
            frange_save_phrase = f'{flead[0]:03d}-F{flead[-1]:03d}'
        frange_string = f'Forecast Hour{frange_phrase}'
        frange_save_string = f'F{frange_save_phrase}'
        df = df[df['LEAD_HOURS'].isin(flead)]
    elif isinstance(flead, tuple):
        frange_string = (f'Forecast Hours {flead[0]:02d}'
                         + u'\u2013' + f'{flead[1]:02d}')
        frange_save_string = f'F{flead[0]:03d}-F{flead[1]:03d}'
        df = df[
            (df['LEAD_HOURS'] >= flead[0]) & (df['LEAD_HOURS'] <= flead[1])
        ]
    elif isinstance(flead, np.int):
        frange_string = f'Forecast Hour {flead:02d}'
        frange_save_string = f'F{flead:03d}'
        df = df[df['LEAD_HOURS'] == flead]
    else:
        error_string = (
            f"FATAL ERROR: Invalid forecast lead: \'{flead}\'\nPlease check settings for"
            + f" forecast leads."
        )
        logger.error(error_string)
        raise ValueError(error_string)
    return df, frange_string, frange_save_string

def filter_by_width(logger, df, interp_pts):
    if interp_pts and '' not in interp_pts:
        interp_shape = list(df['INTERP_MTHD'])[0]
        if 'SQUARE' in interp_shape:
            widths = [int(np.sqrt(float(p))) for p in interp_pts]
        elif 'CIRCLE' in interp_shape:
            widths = [int(np.sqrt(float(p)+4)) for p in interp_pts]
        elif np.all([int(p) == 1 for p in interp_pts]):
            widths = [1 for p in interp_pts]
        else:
            error_string = (
                f"FATAL ERROR: Unknown INTERP_MTHD used to compute INTERP_PNTS: {interp_shape}."
                + f" Check the INTERP_MTHD column in your METplus stats files."
                + f" INTERP_MTHD must have either \"SQUARE\" or \"CIRCLE\""
                + f" in the name."
            )
            logger.error(error_string)
            raise ValueError(error_string)
        if isinstance(interp_pts, list):
            if len(interp_pts) <= 8:
                if len(interp_pts) > 1:
                    interp_pts_phrase = 's '+', '.join([str(p) for p in widths])
                else:
                    interp_pts_phrase = ' '+', '.join([str(p) for p in widths])
                interp_pts_save_phrase = '-'.join([str(p) for p in widths])
            else:
                interp_pts_phrase = f's {widths[0]}'+u'\u2013'+f'{widths[-1]}'
                interp_pts_save_phrase = f'{widths[0]}-{widths[-1]}'
            interp_pts_string = f'(Width{interp_pts_phrase})'
            interp_pts_save_string = f'width{interp_pts_save_phrase}'
            df = df[df['INTERP_PNTS'].isin(interp_pts)]
        elif isinstance(interp_pts, np.int):
            interp_pts_string = f'(Width {widths:d})'
            interp_pts_save_string = f'width{widths:d}'
            df = df[df['INTERP_PNTS'] == widths]
        else:
            error_string = (
                f"FATAL ERROR: Invalid interpolation points entry: \'{interp_pts}\'\n"
                + f"Please check settings for interpolation points."
            )
            logger.error(error_string)
            raise ValueError(error_string)
    else:
        interp_pts_string = ''
        interp_pts_save_string = ''
    return df, interp_pts_string, interp_pts_save_string

def format_thresh(thresh):
   """! Format thresholds for file naming
      Args:
         thresh         - string of the threshold(s)

      Return:
         thresh_symbol  - string of the threshold(s)
                          with symbols
         thresh_letters - string of the threshold(s)
                          with letters
   """
   thresh_list = thresh.split(' ')
   thresh_symbol = ''
   thresh_letter = ''
   for thresh in thresh_list:
      if thresh == '':
         continue
      thresh_value = thresh
      for opt in ['>=', '>', '==', '!=', '<=', '<',
                  'ge', 'gt', 'eq', 'ne', 'le', 'lt']:
         if opt in thresh_value:
            thresh_opt = opt
            thresh_value = thresh_value.replace(opt, '')
      if thresh_opt in ['>', 'gt']:
         thresh_symbol+='>'+thresh_value
         thresh_letter+='gt'+thresh_value
      elif thresh_opt in ['>=', 'ge']:
         thresh_symbol+='>='+thresh_value
         thresh_letter+='ge'+thresh_value
      elif thresh_opt in ['<', 'lt']:
         thresh_symbol+='<'+thresh_value
         thresh_letter+='lt'+thresh_value
      elif thresh_opt in ['<=', 'le']:
         thresh_symbol+='<='+thresh_value
         thresh_letter+='le'+thresh_value
      elif thresh_opt in ['==', 'eq']:
         thresh_symbol+='=='+thresh_value
         thresh_letter+='eq'+thresh_value
      elif thresh_opt in ['!=', 'ne']:
         thresh_symbol+='!='+thresh_value
         thresh_letter+='ne'+thresh_value
   return thresh_symbol, thresh_letter

def get_clevels(data, spacing):
   """! Get contour levels for plotting differences
        or bias (centered on 0)
           Args:
              data    - array of data to be contoured
              spacing - float for spacing for power function,
                        value of 1.0 gives evenly spaced
                        contour intervals
           Returns:
              clevels - array of contour levels
   """
   if np.abs(np.nanmin(data)) > np.nanmax(data):
      cmax = np.abs(np.nanmin(data))
      cmin = np.nanmin(data)
   else:
      cmax = np.nanmax(data)
      cmin = -1 * np.nanmax(data)
   if cmax > 100:
      cmax = cmax - (cmax * 0.2)
      cmin = cmin + (cmin * 0.2)
   elif cmax > 10:
      cmax = cmax - (cmax * 0.1)
      cmin = cmin + (cmin * 0.1)
   if cmax > 1:
      cmin = round(cmin-1,0)
      cmax = round(cmax+1,0)
   else:
      cmin = round(cmin-.1,1)
      cmax = round(cmax+.1,1)
   steps = 6
   span = cmax
   dx = 1. / (steps-1)
   pos = np.array([0 + (i*dx)**spacing*span for i in range(steps)],
                   dtype=float)
   neg = np.array(pos[1:], dtype=float) * -1
   clevels = np.append(neg[::-1], pos)
   return clevels

def get_domain_info(df, domain_translator):
    domain = df['VX_MASK'].tolist()[0]
    if domain in list(domain_translator.keys()):
        domain_string = domain_translator[domain]['long_name']
        domain_save_string = domain_translator[domain]['save_name']
    else:
        domain_string = domain
        domain_save_string = domain
    return domain_string, domain_save_string

def get_level_info(verif_type, level, var_long_name_key, var_savename):
    if str(level).upper() in ['CEILING', 'TOTAL', 'PBL']:
        if str(level).upper() == 'CEILING':
            level_string = ''
            level_savename = 'L0'
        elif str(level).upper() == 'TOTAL':
            level_string = 'Total '
            level_savename = 'L0'
        elif str(level).upper() == 'PBL':
            level_string = ''
            level_savename = 'L0'
    elif str(verif_type).lower() in ['pres', 'upper_air', 'raob'] or 'P' in str(level):
        if 'P' in str(level):
            if str(level).upper() == 'P90-0':
                level_string = f'Mixed-Layer '
                level_savename = f'L90'
            else:
                level_num = level.replace('P', '')
                level_string = f'{level_num} hPa '
                level_savename = f'{level}'
        elif str(level).upper() == 'L0':
            level_string = f'Surface-Based '
            level_savename = f'{level}'
        else:
            level_string = ''
            level_savename = f'{level}'
    elif str(verif_type).lower() in ['sfc', 'conus_sfc', 'polar_sfc', 'metar']:
        if 'Z' in str(level):
            if str(level).upper() == 'Z0':
                if str(var_long_name_key).upper() in ['MLSP', 'MSLET', 'MSLMA', 'PRMSL']:
                    level_string = ''
                    level_savename = f'{level}'
                else:
                    level_string = 'Surface '
                    level_savename = f'{level}'
            else:
                level_num = level.replace('Z', '')
                if var_savename in ['TSOIL', 'SOILW']:
                    level_string = f'{level_num}-cm '
                    level_savename = f'{level_num}CM'
                else:
                    level_string = f'{level_num}-m '
                    level_savename = f'{level}'
        elif 'L' in str(level) or 'A' in str(level):
            level_string = ''
            level_savename = f'{level}'
        else:
            level_string = f'{level} '
            level_savename = f'{level}'
    elif str(verif_type).lower() in ['ccpa', 'mrms']:
        if 'A' in str(level):
            level_num = level.replace('A', '')
            level_string = f'{level_num}-hour '
            level_savename = f'A{level_num.zfill(2)}'
        else:
            level_string = f''
            level_savename = f'{level}'
    else:
        level_string = f'{level} '
        level_savename = f'{level}'
    return level_string, level_savename

def get_MCTC_cols_for_sum(n_cats, i_vals, ctc_metric_name):
    cols = []
    if ctc_metric_name.lower() in ['fy_oy','a']:
        for i_val in i_vals:
            F_num = int(i_val)
            O_num = int(i_val)
            cols.append([f'F{F_num}_O{O_num}'])
    elif ctc_metric_name.lower() in ['fy_on','b']:
        for n, i_val in enumerate(i_vals):
            cols_for_sum = []
            for ii in np.arange(n_cats[n], dtype='int'):
                if int(i_val) != int(ii):
                    F_num = int(i_val)
                    O_num = int(ii)
                    cols_for_sum.append(f'F{F_num}_O{O_num}')
            cols.append(cols_for_sum)
    elif ctc_metric_name.lower() in ['fn_oy','c']:
        for n, i_val in enumerate(i_vals):
            cols_for_sum = []
            for ii in np.arange(n_cats[n], dtype='int'):
                if int(i_val) != int(ii):
                    F_num = int(ii)
                    O_num = int(i_val)
                    cols_for_sum.append(f'F{F_num}_O{O_num}')
            cols.append(cols_for_sum)
    else:
        print(f"ctc_metric_name, {ctc_metric_name} is not permitted.")
        sys.exit(1)
    return cols

def get_memory_usage():
    total_memory, used_memory, free_memory = map(
        int, 
        os.popen('free -t -m').readlines()[-1].split()[1:]
    )
    return ' '.join((
        "RAM memory % used:", 
        str(round((used_memory/total_memory) * 100, 2))
    ))

def get_model_settings(model_list, model_colors):
    models_renamed = []
    count_renamed = 1
    for requested_model in model_list:
        if requested_model in model_colors.model_alias:
            requested_model = (
                model_colors.model_alias[requested_model]['settings_key']
            )
        if requested_model in model_settings:
            models_renamed.append(requested_model)
        else:
            models_renamed.append('model'+str(count_renamed))
            count_renamed+=1
    models_renamed = np.array(models_renamed)
    # Check that there are no repeated colors
    temp_colors = [
        model_colors.get_color_dict(name)['color'] for name in models_renamed
    ]
    colors_corrected=False
    loop_count=0
    while not colors_corrected and loop_count < 10:
        unique, counts = np.unique(temp_colors, return_counts=True)
        repeated_colors = [u for i, u in enumerate(unique) if counts[i] > 1]
        if repeated_colors:
            for c in repeated_colors:
                models_sharing_colors = models_renamed[
                    np.array(temp_colors)==c
                ]
                if np.flatnonzero(np.core.defchararray.find(
                        models_sharing_colors, 'model')!=-1):
                    need_to_rename = models_sharing_colors[np.flatnonzero(
                        np.core.defchararray.find(
                            models_sharing_colors, 'model'
                        )!=-1)[0]
                    ]
                else:
                    continue
                models_renamed[models_renamed==need_to_rename] = (
                    'model'+str(count_renamed)
                )
                count_renamed+=1
            temp_colors = [
                model_colors.get_color_dict(name)['color'] 
                for name in models_renamed
            ]
            loop_count+=1
        else:
            colors_corrected = True
    mod_setting_dicts = [
        model_colors.get_color_dict(name) for name in models_renamed
    ]
    return mod_settings_dicts

def get_model_stats_key(model_alias_dict, requested_model):
    if requested_model not in model_alias_dict:
        return requested_model
    else:
        stats_key = model_alias_dict[requested_model]['stats_key']
        if not stats_key:
            return requested_model
        else:
            return stats_key

def get_name_for_listed_items(listed_items, joinchars, prechars, postchars, prechars_last, postchars_last):
    new_items = []
    if len(listed_items) == 1:
        return f"{prechars}{listed_items[0]}{postchars}"
    elif len(listed_items) == 2:
        return (f"{prechars}{listed_items[0]}{postchars} {prechars_last}"
                + f"{prechars}{listed_items[1]}{postchars}{postchars_last}")
    else:
        for i, item in enumerate(listed_items):
            if i == 0:
                start = item
            elif i == (len(listed_items)-1):
                if int(item) == int(listed_items[i-1]):
                    new_items.append(f"{prechars_last}{prechars}{start}{postchars}{postchars_last}")
                elif int(item) != int(listed_items[i-1])+1:
                    if int(start) != int(listed_items[i-1]):
                        new_items.append(f"{prechars}{start}-{listed_items[i-1]}{postchars}")
                        new_items.append(f"{prechars_last}{prechars}{item}{postchars}{postchars_last}")
                    else:
                        new_items.append(f"{prechars}{listed_items[i-1]}{postchars}")
                        new_items.append(f"{prechars_last}{prechars}{item}{postchars}{postchars_last}")
                elif int(start) == int(listed_items[0]):
                    new_items.append(f"{prechars}{start}-{item}{postchars}")
                else:
                    new_items.append(f"{prechars_last}{prechars}{start}-{item}{postchars}{postchars_last}")
            elif int(item) != int(listed_items[i-1])+1:
                if int(start) == int(listed_items[i-1]):
                    new_items.append(f"{prechars}{start}{postchars}")
                else:
                    new_items.append(
                        f"{prechars}{start}-{listed_items[i-1]}{postchars}"
                    )
                start=item
    return joinchars.join(new_items)

def get_pivot_tables(df_aggregated, metric1_name, metric2_name, 
                     sample_equalization, keep_shared_events_only, date_type, 
                     confidence_intervals, plot_type, aggregate_dates_by='', 
                     colname='MODEL'):
    if plot_type == 'timeseries':
        if aggregate_dates_by:
            if aggregate_dates_by in ['m','month']:
                df_aggregated['MONTH'] = df_aggregated[
                    str(date_type).upper()
                ].dt.strftime('%Y%m')
                index_colname = 'MONTH'
            elif aggregate_dates_by in ['Y','year']:
                df_aggregated['YEAR'] = df_aggregated[
                    str(date_type).upper()
                ].dt.strftime('%Y')
                index_colname = 'YEAR'
            else:
                raise ValueError(
                    f"Unknown value for aggregate_dates_by: "
                    + "{aggregate_dates_by}"
                )
        else:
            index_colname = str(date_type).upper()
    elif plot_type == 'fhrmean':
        index_colname = 'LEAD_HOURS'
    pivot_metric1 = pd.pivot_table(
        df_aggregated, values=str(metric1_name).upper(), columns=colname, 
        index=index_colname
    )
    if sample_equalization:
        pivot_counts = pd.pivot_table(
            df_aggregated, values='COUNTS', columns=colname,
            index=index_colname
        )
    if keep_shared_events_only:
        pivot_metric1 = pivot_metric1.dropna() 
    if metric2_name is not None:
        pivot_metric2 = pd.pivot_table(
            df_aggregated, values=str(metric2_name).upper(), columns=colname, 
            index=index_colname
        )
        if keep_shared_events_only:
            pivot_metric2 = pivot_metric2.dropna()
    else:
        pivot_metric2 = None
    if confidence_intervals:
        pivot_ci_lower1 = pd.pivot_table(
            df_aggregated, values=str(metric1_name).upper()+'_BLERR',
            columns=colname, index=index_colname
        )
        pivot_ci_upper1 = pd.pivot_table(
            df_aggregated, values=str(metric1_name).upper()+'_BUERR',
            columns=colname, index=index_colname
        )
        if metric2_name is not None:
            pivot_ci_lower2 = pd.pivot_table(
                df_aggregated, values=str(metric2_name).upper()+'_BLERR',
                columns=colname, index=index_colname
            )
            pivot_ci_upper2 = pd.pivot_table(
                df_aggregated, values=str(metric2_name).upper()+'_BUERR',
                columns=colname, index=index_colname
            )
        else:
            pivot_ci_lower2 = None
            pivot_ci_upper2 = None
    else:
        pivot_ci_lower1 = None
        pivot_ci_upper1 = None
        pivot_ci_lower2 = None
        pivot_ci_upper2 = None
    return (
        pivot_metric1, pivot_metric2, pivot_ci_lower1, pivot_ci_upper1, 
        pivot_ci_lower2, pivot_ci_upper2
    )

def get_stat_file_base_columns(met_version):
   """! Get the standard MET .stat file columns based on
        version number

           Args:
              met_version            - string of MET version
                                       number being used to
                                       run stat_analysis

           Returns:
              stat_file_base_columns - list of the standard
                                       columns shared among the 
                                       different line types
   """
   met_version = float(met_version)
   if met_version < 8.1:
      stat_file_base_columns = [
         'VERSION', 'MODEL', 'DESC', 'FCST_LEAD', 'FCST_VALID_BEG',
         'FCST_VALID_END', 'OBS_LEAD', 'OBS_VALID_BEG', 'OBS_VALID_END',
         'FCST_VAR', 'FCST_LEV', 'OBS_VAR', 'OBS_LEV', 'OBTYPE', 'VX_MASK',
         'INTERP_MTHD', 'INTERP_PNTS', 'FCST_THRESH', 'OBS_THRESH', 
         'COV_THRESH', 'ALPHA', 'LINE_TYPE'
      ]
   else:
      stat_file_base_columns = [
         'VERSION', 'MODEL', 'DESC', 'FCST_LEAD', 'FCST_VALID_BEG', 
         'FCST_VALID_END', 'OBS_LEAD', 'OBS_VALID_BEG', 'OBS_VALID_END', 
         'FCST_VAR', 'FCST_UNITS', 'FCST_LEV', 'OBS_VAR', 'OBS_UNITS', 
         'OBS_LEV', 'OBTYPE', 'VX_MASK', 'INTERP_MTHD', 'INTERP_PNTS',
         'FCST_THRESH', 'OBS_THRESH', 'COV_THRESH', 'ALPHA', 'LINE_TYPE'
      ]
   return stat_file_base_columns

def get_stat_file_line_type_columns(logger, met_version, line_type, 
                                    stat_file_base_columns, fpath):
   """! Get the MET .stat file columns for line type based on 
      version number
         Args:
            met_version - string of MET version number 
                          being used to run stat_analysis
            line_type   - string of the line type of the MET
                          .stat file being read
         Returns:
            stat_file_line_type_columns - list of the line
                                          type columns
   """
   met_version = float(met_version)
   if line_type == 'SL1L2':
      if met_version >= 6.0:
         stat_file_line_type_columns = [
            'TOTAL', 'FBAR', 'OBAR', 'FOBAR', 'FFBAR', 'OOBAR', 'MAE'
         ]
   elif line_type == 'SAL1L2':
      if met_version >= 6.0:
         stat_file_line_type_columns = [
            'TOTAL', 'FABAR', 'OABAR', 'FOABAR', 'FFABAR', 'OOABAR', 'MAE'
         ]
   elif line_type == 'VL1L2':
      if met_version >= 12.0:
         stat_file_line_type_columns = [
            'TOTAL', 'UFBAR', 'VFBAR', 'UOBAR', 'VOBAR', 'UVFOBAR',
            'UVFFBAR', 'UVOOBAR', 'F_SPEED_BAR', 'O_SPEED_BAR', 'DIR_ME',
            'DIR_MAE', 'DIR_MSE'
         ]
      elif met_version >= 7.0:
         stat_file_line_type_columns = [
            'TOTAL', 'UFBAR', 'VFBAR', 'UOBAR', 'VOBAR', 'UVFOBAR',
            'UVFFBAR', 'UVOOBAR', 'F_SPEED_BAR', 'O_SPEED_BAR'
         ]
      elif met_version <= 6.1:
         stat_file_line_type_columns = [
            'TOTAL', 'UFBAR', 'VFBAR', 'UOBAR', 'VOBAR', 'UVFOBAR',
            'UVFFBAR', 'UVOOBAR'
         ]
   elif line_type == 'VAL1L2':
      if met_version >= 11.0:
         stat_file_line_type_columns = [
            'TOTAL', 'UFABAR', 'VFABAR', 'UOABAR', 'VOABAR', 'UVFOABAR', 
            'UVFFABAR', 'UVOOABAR', 'FA_SPEED_BAR', 'OA_SPEED_BAR'
         ]
      elif met_version >= 6.0:
         stat_file_line_type_columns = [
            'TOTAL', 'UFABAR', 'VFABAR', 'UOABAR', 'VOABAR', 'UVFOABAR', 
            'UVFFABAR', 'UVOOABAR'
         ]
   elif line_type == 'VCNT':
      if met_version >= 11.0:
         stat_file_line_type_columns = [
            'TOTAL', 'FBAR', 'FBAR_BCL', 'FBAR_BCU', 'OBAR', 'OBAR_BCL', 
            'OBAR_BCU', 'FS_RMS', 'FS_RMS_BCL', 'FS_RMS_BCU', 'OS_RMS',
            'OS_RMS_BCL', 'OS_RMS_BCU', 'MSVE', 'MSVE_BCL', 'MSVE_BCU',
            'RMSVE', 'RMSVE_BCL', 'RMSVE_BCU', 'FSTDEV', 'FSTDEV_BCL',
            'FSTDEV_BCU', 'OSTDEV', 'OSTDEV_BCL', 'OSTDEV_BCU', 'FDIR', 
            'FDIR_BCL', 'FDIR_BCU', 'ODIR', 'ODIR_BCL', 'ODIR_BCU', 
            'FBAR_SPEED', 'FBAR_SPEED_BCL', 'FBAR_SPEED_BCU', 'OBAR_SPEED', 
            'OBAR_SPEED_BCL', 'OBAR_SPEED_BCU', 'VDIFF_SPEED', 
            'VDIFF_SPEED_BCL', 'VDIFF_SPEED_BCU', 'VDIFF_DIR',
            'VDIFF_DIR_BCL', 'VDIFF_DIR_BCU', 'SPEED_ERR', 'SPEED_ERR_BCL',
            'SPEED_ERR_BCU', 'SPEED_ABSERR', 'SPEED_ABSERR_BCL',
            'SPEED_ABSERR_BCU', 'DIR_ERR', 'DIR_ERR_BCL', 'DIR_ERR_BCU',
            'DIR_ABSERR', 'DIR_ABSERR_BCL', 'DIR_ABSERR_BCU', 'ANOM_CORR',
            'ANOM_CORR_NCL', 'ANOM_CORR_NCU', 'ANOM_CORR_BCL', 'ANOM_CORR_BCU',
            'ANOM_CORR_UNCNT', 'ANOM_CORR_UNCNTR_BCL', 'ANOM_CORR_UNCNTR_BCU'
         ]
      elif met_version >= 7.0:
         stat_file_line_type_columns = [
            'TOTAL', 'FBAR', 'FBAR_NCL', 'FBAR_NCU', 'OBAR', 'OBAR_NCL', 
            'OBAR_NCU', 'FS_RMS', 'FS_RMS_NCL', 'FS_RMS_NCU', 'OS_RMS',
            'OS_RMS_NCL', 'OS_RMS_NCU', 'MSVE', 'MSVE_NCL', 'MSVE_NCU',
            'RMSVE', 'RMSVE_NCL', 'RMSVE_NCU', 'FSTDEV', 'FSTDEV_NCL',
            'FSTDEV_NCU', 'OSTDEV', 'OSTDEV_NCL', 'OSTDEV_NCU', 'FDIR', 
            'FDIR_NCL', 'FDIR_NCU', 'ODIR', 'ODIR_NCL', 'ODIR_NCU', 
            'FBAR_SPEED', 'FBAR_SPEED_NCL', 'FBAR_SPEED_NCU', 'OBAR_SPEED', 
            'OBAR_SPEED_NCL', 'OBAR_SPEED_NCU', 'VDIFF_SPEED', 
            'VDIFF_SPEED_NCL', 'VDIFF_SPEED_NCU', 'VDIFF_DIR',
            'VDIFF_DIR_NCL', 'VDIFF_DIR_NCU', 'SPEED_ERR', 'SPEED_ERR_NCL',
            'SPEED_ERR_NCU', 'SPEED_ABSERR', 'SPEED_ABSERR_NCL',
            'SPEED_ABSERR_NCU', 'DIR_ERR', 'DIR_ERR_NCL', 'DIR_ERR_NCU',
            'DIR_ABSERR', 'DIR_ABSERR_NCL', 'DIR_ABSERR_NCU'
         ]
      else:
         logger.error("FATAL ERROR: VCNT is not a valid LINE_TYPE in METV"+met_version)
         exit(1)
   elif line_type == 'CTC':
      if met_version >= 11.0:
         stat_file_line_type_columns = [
            'TOTAL', 'FY_OY', 'FY_ON', 'FN_OY', 'FN_ON', 'EC_VALUE'
         ]
      elif met_version >= 6.0:
          stat_file_line_type_columns = [
            'TOTAL', 'FY_OY', 'FY_ON', 'FN_OY', 'FN_ON'
         ]
   elif line_type == 'NBRCTC':
       if met_version >= 6.0:
          stat_file_line_type_columns = [
            'TOTAL', 'FY_OY', 'FY_ON', 'FN_OY', 'FN_ON'
         ]
   elif line_type == 'NBRCNT':
      if met_version >= 6.0:
         stat_file_line_type_columns = [
            'TOTAL', 'FBS', 'FBS_BCL', 'FBS_BCU', 'FSS', 'FSS_BCL', 'FSS_BCU',
            'AFSS', 'AFSS_BCL', 'AFSS_BCU', 'UFSS', 'UFSS_BCL', 'UFSS_BCU',
            'F_RATE', 'F_RATE_BCL', 'F_RATE_BCU',
            'O_RATE', 'O_RATE_BCL', 'O_RATE_BCU'
         ]
   elif line_type == 'ECNT':
      if met_version >= 12.0:
         stat_file_line_type_columns = [
             'TOTAL', 'N_ENS', 'CRPS', 'CRPSS', 'IGN', 'ME', 'RMSE', 'SPREAD',
             'ME_OERR', 'RMSE_OERR', 'SPREAD_OERR', 'SPREAD_PLUS_OERR',
             'CRPSCL', 'CRPS_EMP', 'CRPSCL_EMP', 'CRPSS_EMP',
             'CRPS_EMP_FAIR', 'SPREAD_MD', 'MAE', 'MAE_OERR', 'BIAS_RATIO',
             'N_GE_OBS', 'ME_GE_OBS', 'N_LT_OBS', 'ME_LT_OBS', 'IGN_CONV_OERR',
             'IGN_CORR_OERR'
         ] 
      elif met_version >= 11.0:
         stat_file_line_type_columns = [
             'TOTAL', 'N_ENS', 'CRPS', 'CRPSS', 'IGN', 'ME', 'RMSE', 'SPREAD',
             'ME_OERR', 'RMSE_OERR', 'SPREAD_OERR', 'SPREAD_PLUS_OERR',
             'CRPSCL', 'CRPS_EMP', 'CRPSCL_EMP', 'CRPSS_EMP',
             'CRPS_EMP_FAIR', 'SPREAD_MD', 'MAE', 'MAE_OERR', 'BIAS_RATIO',
             'N_GE_OBS', 'ME_GE_OBS', 'N_LT_OBS', 'ME_LT_OBS'
         ] 
      else:
         stat_file_line_type_columns = [
            'TOTAL', 'N_ENS', 'CRPS', 'CRPSS', 'IGN', 'ME', 'RMSE', 'SPREAD',
            'ME_OERR', 'RMSE_OERR', 'SPREAD_OERR', 'SPREAD_PLUS_OERR',
            'CRPSCL', 'CRPS_EMP', 'CRPSCL_EMP', 'CRPSS_EMP'
         ]
   elif line_type == 'PSTD':
      if met_version >= 6.0:
         stat_file_line_type_columns = [
            'TOTAL', 'N_THRESH', 'BASER', 'BASER_NCL', 'BASER_NCU', 'RELIABILITY',
            'RESOLUTION', 'UNCERTAINTY', 'ROC_AUC', 'BRIER', 'BRIER_NCL', 'BRIER_NCU',
            'BRIERCL', 'BRIERCL_NCL', 'BRIERCL_NCU', 'BSS', 'BSS_SMPL',
            'THRESH_1', 'THRESH_2', 'THRESH_3', 'THRESH_4', 'THRESH_5', 'THRESH_6',
            'THRESH_7', 'THRESH_8', 'THRESH_9', 'THRESH_10', 'THRESH_11'
         ]
   elif line_type == 'MCTC':
      if met_version >= 11.0:
         # need to pull in stat_file_og_columns and fname as args!
         stat_file_line_type_columns_start = ['TOTAL', 'N_CAT']
         stat_file_all_columns_start = np.concatenate((
            stat_file_base_columns, stat_file_line_type_columns_start
         ))
         df_read_tmp = pd.read_csv(
            fpath, delim_whitespace=True, header=None, skiprows=1, dtype=str
         )
         categs = np.arange(int(
            df_read_tmp[
                np.argwhere(stat_file_all_columns_start=='N_CAT')[0]
            ].max()
         ))
         variable_columns = []
         for Fcateg in categs:
            for Ocateg in categs:
               variable_columns.append(f'F{Fcateg}_O{Ocateg}')
         stat_file_line_type_columns = np.concatenate((
            stat_file_line_type_columns_start,
            variable_columns,
            ['EC_VALUE']
         ))
      elif met_version >= 6.0:
         # need to pull in stat_file_og_columns and fname as args!
         stat_file_line_type_columns_start = ['TOTAL', 'N_CAT']
         stat_file_all_columns_start = np.concatenate((
            stat_file_og_columns, stat_file_line_type_columns_start
         ))
         df_read_tmp = pd.read_csv(
            fname, delim_whitespace=True, header=None, skiprows=1, dtype=str
         )
         categs = np.arange(int(
            df_read_tmp[
                np.argwhere(stat_file_all_columns_start=='N_CAT')[0]
            ].max()
         ))
         variable_columns = []
         for Fcateg in categs:
            for Ocateg in categs:
               variable_columns.append(f'F{Fcat}_O{Ocat}')
         stat_file_line_type_columns = np.concatenate((
            stat_file_line_type_columns_start,
            variable_columns
         ))
   return stat_file_line_type_columns

def get_stat_plot_name(logger, stat):
   """! Get the formalized name of the statistic being plotted
      Args:
         stat           - string of the simple statistic
                          name being plotted
      Returns:
         stat_plot_name - string of the formal statistic
                          name being plotted
   """
   if stat == 'me':
      stat_plot_name = 'Mean Error (i.e., Bias)'
   elif stat == 'rmse':
      stat_plot_name = 'Root Mean Square Error'
   elif stat == 'bcrmse':
      stat_plot_name = 'Bias-Corrected Root Mean Square Error'
   elif stat == 'msess':
      stat_plot_name = "Murphy's Mean Square Error Skill Score"
   elif stat == 'rsd':
      stat_plot_name = 'Ratio of the Standard Deviation'
   elif stat == 'rmse_md':
      stat_plot_name = 'Root Mean Square Error from Mean Error'
   elif stat == 'rmse_pv':
      stat_plot_name = 'Root Mean Square Error from Pattern Variation'
   elif stat == 'pcor':
      stat_plot_name = 'Pattern Correlation'
   elif stat == 'acc':
      stat_plot_name = 'Anomaly Correlation Coefficient'
   elif stat == 'fbar':
      stat_plot_name = 'Forecast Mean'
   elif stat == 'obar':
      stat_plot_name = 'Observation Mean'
   elif stat == 'fbar_obar':
      stat_plot_name = 'Forecast and Observation Mean'
   elif stat == 'fss':
      stat_plot_name = 'Fractions Skill Score'
   elif stat == 'afss':
      stat_plot_name = 'Asymptotic Fractions Skill Score'
   elif stat == 'ufss':
      stat_plot_name = 'Uniform Fractions Skill Score'
   elif stat == 'speed_err':
      stat_plot_name = (
         'Difference in Average FCST and OBS Wind Vector Speeds'
      )
   elif stat == 'dir_err':
      stat_plot_name = (
         'Difference in Average FCST and OBS Wind Vector Direction'
      )
   elif stat == 'rmsve':
      stat_plot_name = 'Root Mean Square Difference Vector Error'
   elif stat == 'vdiff_speed':
      stat_plot_name = 'Difference Vector Speed'
   elif stat == 'vdiff_dir':
      stat_plot_name = 'Difference Vector Direction'
   elif stat == 'fbar_obar_speed':
      stat_plot_name = 'Average Wind Vector Speed'
   elif stat == 'fbar_obar_dir':
      stat_plot_name = 'Average Wind Vector Direction'
   elif stat == 'fbar_speed':
      stat_plot_name = 'Average Forecast Wind Vector Speed'
   elif stat == 'fbar_dir':
      stat_plot_name = 'Average Forecast Wind Vector Direction'
   elif stat == 'orate':
      stat_plot_name = 'Observation Rate'
   elif stat == 'baser':
      stat_plot_name = 'Base Rate'
   elif stat == 'frate':
      stat_plot_name = 'Forecast Rate'
   elif stat == 'orate_frate':
      stat_plot_name = 'Observation and Forecast Rates'
   elif stat == 'baser_frate':
      stat_plot_name = 'Base and Forecast Rates'
   elif stat == 'accuracy':
      stat_plot_name = 'Accuracy'
   elif stat == 'fbias':
      stat_plot_name = 'Frequency Bias'
   elif stat == 'pod':
      stat_plot_name = 'Probability of Detection'
   elif stat == 'hrate':
      stat_plot_name = 'Hit Rate'
   elif stat == 'pofd':
      stat_plot_name = 'Probability of False Detection'
   elif stat == 'farate':
      stat_plot_name = 'False Alarm Rate'
   elif stat == 'podn':
      stat_plot_name = 'Probability of Detection of the Non-Event'
   elif stat == 'faratio':
      stat_plot_name = 'False Alarm Ratio'
   elif stat == 'sratio':
      stat_plot_name = 'Success Ratio (1-FAR)'
   elif stat == 'csi':
      stat_plot_name = 'Critical Success Index'
   elif stat == 'ts':
      stat_plot_name = 'Threat Score'
   elif stat == 'gss':
      stat_plot_name = 'Gilbert Skill Score'
   elif stat == 'ets':
      stat_plot_name = 'Equitable Threat Score'
   elif stat == 'hk':
      stat_plot_name = 'Hanssen-Kuipers Discriminant'
   elif stat == 'tss':
      stat_plot_name = 'True Skill Score'
   elif stat == 'pss':
      stat_plot_name = 'Peirce Skill Score'
   elif stat == 'hss':
      stat_plot_name = 'Heidke Skill Score'
   elif stat == 'crps':
      stat_plot_name = 'CRPS'
   elif stat == 'crpss':
      stat_plot_name = 'CRPSS'
   elif stat == 'spread':
      stat_plot_name = 'Spread'
   elif stat == 'me':
      stat_plot_name = 'Mean Error (Bias)'
   elif stat == 'mae':
      stat_plot_name = 'Mean Absolute Error'
   elif stat == 'bs':
      stat_plot_name = 'Brier Score'
   elif stat == 'roc_area':
      stat_plot_name = 'ROC Area'
   elif stat == 'bss':
      stat_plot_name = 'Brier Skill Score'
   elif stat == 'bss_smpl':
      stat_plot_name = 'Brier Skill Score'
   else:
      logger.error("FATAL ERROR: "+stat+" is not a valid option")
      exit(1)
   return stat_plot_name

def get_var_info(df):
    var_savename = df['FCST_VAR'].tolist()[0]
    if 'APCP' in var_savename.upper():
        var_savename = 'APCP'
    elif any(field in var_savename.upper() for field in ['ASNOW','SNOD']):
        var_savename = 'ASNOW'
    elif str(df['OBS_VAR'].tolist()[0]).upper() in ['HPBL']:
        var_savename = 'HPBL'
    elif str(df['OBS_VAR'].tolist()[0]).upper() in ['MSLET','MSLMA','PRMSL']:
        var_savename = 'MSLET'
    return var_savename

def process_models(logger, df, model_list):
    cols_to_keep = [
        str(model) 
        in df['MODEL'].tolist() 
        for model in model_list
    ]
    models_removed = [
        str(m) 
        for (m, keep) in zip(model_list, cols_to_keep) if not keep
    ]
    models_removed_string = ', '.join(models_removed)
    model_list = [
        str(m) 
        for (m, keep) in zip(model_list, cols_to_keep) if keep
    ]
    if not all(cols_to_keep):
        if not any(
                group_name in str(models_removed_string) 
                for group_name in ["group", "set"]
            ):
            logger.warning(
                f"{models_removed_string} data were not found and will not be"
                + f" plotted."
            )
    return df, model_list

def process_stats(logger, df, df_groups, model_list, metric1_name, metric2_name, 
                  metrics_using_var_units, confidence_intervals, date_type,
                  line_type, plot_type, bs_method, bs_nrep, bs_min_samp, ci_lev,
                  reference, sample_equalization=True, keep_shared_events_only=True, 
                  delete_intermed_data=False):
    df_aggregated = aggregate_stats(
        df_groups, model_list, date_type, line_type, plot_type, 
        sample_equalization=sample_equalization,
        keep_shared_events_only=keep_shared_events_only,
        delete_intermed_data=delete_intermed_data
    )
    
    if df_aggregated.empty:
        logger.warning(f"Empty Dataframe. Continuing onto next plot...")
        plt.close(num)
        logger.info("========================================")
        return None
    
    metrics_using_var_units = [
        'BCRMSE','RMSE','BIAS','ME','FBAR','OBAR','MAE','FBAR_OBAR',
        'SPEED_ERR','DIR_ERR','RMSVE','VDIFF_SPEED','VDIF_DIR',
        'FBAR_OBAR_SPEED','FBAR_OBAR_DIR','FBAR_SPEED','FBAR_DIR'
    ]
    #
    units = df['FCST_UNITS'].tolist()[0]
    coef, const = (None, None)
    unit_convert = False
    if units in reference.unit_conversions:
        unit_convert = True
        var_long_name_key = df['FCST_VAR'].tolist()[0]
        if str(var_long_name_key).upper() == 'HGT':
            if str(df['OBS_VAR'].tolist()[0]).upper() in ['CEILING']:
                if units in ['m','gpm']:
                    units = 'gpm'
            elif str(df['OBS_VAR'].tolist()[0]).upper() in ['HPBL']:
                unit_convert = False
            elif str(df['OBS_VAR'].tolist()[0]).upper() in ['HGT']:
                unit_convert = False
        elif any(field in str(var_long_name_key).upper() for field in ['WEASD', 'SNOD', 'ASNOW']):
            if units in ['m']:
                units = 'm_snow'
        if unit_convert:
            if metric2_name is not None:
                if (str(metric1_name).upper() in metrics_using_var_units
                        and str(metric2_name).upper() in metrics_using_var_units):
                    coef, const = (
                        reference.unit_conversions[units]['formula'](
                            None,
                            return_terms=True
                        )
                    )
            elif str(metric1_name).upper() in metrics_using_var_units:
                coef, const = (
                    reference.unit_conversions[units]['formula'](
                        None,
                        return_terms=True
                    )
                )
    # Calculate desired metric
    metric_long_names = []
    for stat in [metric1_name, metric2_name]:
        if stat:
            stat_output = calculate_stat(
                logger, df_aggregated, str(stat).lower(), [coef, const]
            )
            df_aggregated[str(stat).upper()] = stat_output[0]
            metric_long_names.append(stat_output[2])
            if confidence_intervals:
                ci_output = df_groups.apply(
                    lambda x: calculate_bootstrap_ci(
                        logger, bs_method, x, str(stat).lower(), bs_nrep,
                        ci_lev, bs_min_samp, [coef, const]
                    )
                )
                if any(ci_output['STATUS'] == 1):
                    logger.warning(f"Failed attempt to compute bootstrap"
                                   + f" confidence intervals.  Sample size"
                                   + f" for one or more groups is too small."
                                   + f" Minimum sample size can be changed"
                                   + f" in settings.py.")
                    logger.warning(f"Confidence intervals will not be"
                                   + f" plotted.")
                    confidence_intervals = False
                    continue
                ci_output = ci_output.reset_index(level=2, drop=True)
                ci_output = (
                    ci_output
                    .reindex(df_aggregated.index)
                    .reindex(ci_output.index)
                )
                df_aggregated[str(stat).upper()+'_BLERR'] = ci_output[
                    'CI_LOWER'
                ].values
                df_aggregated[str(stat).upper()+'_BUERR'] = ci_output[
                    'CI_UPPER'
                ].values
    df_aggregated[str(metric1_name).upper()] = (
        df_aggregated[str(metric1_name).upper()]
    ).astype(float).tolist()
    if metric2_name is not None:
        df_aggregated[str(metric2_name).upper()] = (
            df_aggregated[str(metric2_name).upper()]
        ).astype(float).tolist()
    df_aggregated = df_aggregated[
        df_aggregated.index.isin(model_list, level='MODEL')
    ]
    return df_aggregated, metric_long_names, units, unit_convert

def process_thresh(logger, df, thresh):
    if thresh and '' not in thresh:
        requested_thresh_symbol, requested_thresh_letter = list(
            zip(*[format_thresh(t) for t in thresh])
        )
        symbol_found = False
        for opt in ['>=', '>', '==', '!=', '<=', '<']:
            if any(opt in t for t in requested_thresh_symbol):
                if all(opt in t for t in requested_thresh_symbol):
                    symbol_found = True
                    opt_letter = requested_thresh_letter[0][:2]
                    break
                else:
                    e = ("FATAL ERROR: Threshold operands do not match among all requested"
                         + f" thresholds.")
                    logger.error(e)
                    logger.error("Quitting ...")
                    raise ValueError(e+"\nQuitting ...")
        if not symbol_found:
            e = "FATAL ERROR: None of the requested thresholds contain a valid symbol."
            logger.error(e)
            logger.error("Quitting ...")
            raise ValueError(e+"\nQuitting ...")
        df_thresh_symbol, df_thresh_letter = list(
            zip(*[format_thresh(t) for t in df['FCST_THRESH']])
        )
        df['FCST_THRESH_SYMBOL'] = df_thresh_symbol
        df['FCST_THRESH_VALUE'] = [str(item)[2:] for item in df_thresh_letter]
        requested_thresh_value = [
            str(item)[2:] for item in requested_thresh_letter
        ]
        df = df[df['FCST_THRESH_SYMBOL'].isin(requested_thresh_symbol)]
        thresholds_removed = (
            np.array(requested_thresh_symbol)[
                ~np.isin(requested_thresh_symbol, df['FCST_THRESH_SYMBOL'])
            ]
        )
        requested_thresh_symbol = (
            np.array(requested_thresh_symbol)[
                np.isin(requested_thresh_symbol, df['FCST_THRESH_SYMBOL'])
            ]
        )
        if thresholds_removed.size > 0:
            thresholds_removed_string = ', '.join(thresholds_removed)
            if len(thresholds_removed) > 1:
                warning_string = (f"{thresholds_removed_string} thresholds"
                                  + f" were not found and will not be"
                                  + f" plotted.")
            else:
                warning_string = (f"{thresholds_removed_string} threshold was"
                                  + f" not found and will not be plotted.")
            logger.warning(warning_string)
            logger.warning("Continuing ...")
    return df, opt, opt_letter, requested_thresh_value

def reindex_pivot_tables(pivot_metric1, pivot_metric2, pivot_counts, 
                         pivot_ci_lower1, pivot_ci_upper1, pivot_ci_lower2,
                         pivot_ci_upper2, plot_type, date_hours, metric2_name,
                         sample_equalization, confidence_intervals):
    if plot_type == "timeseries":
        date_hours_incr = np.diff(date_hours)
        if date_hours_incr.size == 0:
            min_incr = 24
        else:
            min_incr = np.min(date_hours_incr)
        incrs = [1,6,12,24]
        incr_idx = np.digitize(min_incr, incrs)
        if incr_idx < 1:
            incr_idx = 1
        incr = incrs[incr_idx-1]
        idx = [
            item 
            for item in daterange(
                date_range[0].replace(hour=np.min(date_hours)), 
                date_range[1].replace(hour=np.max(date_hours)), 
                td(hours=incr)
            )
        ]
        pivot_metric1 = pivot_metric1.reindex(idx, fill_value=np.nan)
        if sample_equalization:
            pivot_counts = pivot_counts.reindex(idx, fill_value=np.nan)
        if confidence_intervals:
            pivot_ci_lower1 = pivot_ci_lower1.reindex(idx, fill_value=np.nan)
            pivot_ci_upper1 = pivot_ci_upper1.reindex(idx, fill_value=np.nan)
        if metric2_name is not None:
            pivot_metric2 = pivot_metric2.reindex(idx, fill_value=np.nan)
            if confidence_intervals:
                pivot_ci_lower2 = pivot_ci_lower2.reindex(idx, fill_value=np.nan)
                pivot_ci_upper2 = pivot_ci_upper2.reindex(idx, fill_value=np.nan)
    elif plot_type == 'fhrmean':
        x_vals_pre = pivot_metric1.index.tolist()
        lead_time_incr = np.diff(x_vals_pre)
        if lead_time_incr.size == 0:
            min_incr = 1
        else:
            min_incr = np.min(lead_time_incr)
        incrs = [1,6,12,24]
        incr_idx = np.digitize(min_incr, incrs)
        if incr_idx < 1:
            incr_idx = 1
        incr = incrs[incr_idx-1]
    else:
        e = f"Plot type (\"{plot_type}\") is not currently supported."
        logger.error(e)
        raise ValueError(e)
    return (
        (
            pivot_metric1, pivot_metric2, pivot_counts, 
            pivot_ci_lower1, pivot_ci_upper1,
            pivot_ci_lower2, pivot_ci_upper2
        ),
        incr)
