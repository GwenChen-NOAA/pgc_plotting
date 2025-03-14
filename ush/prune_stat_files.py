#!/usr/bin/env python3

'''
Program Name: prune_stat_files.py
Contact(s): Marcel Caron
Abstract: This prunes the MET .stat files for the
          specific plotting job to help decrease
          wall time.
'''

import glob
import subprocess
import os
import re
import sys
import numpy as np
from datetime import datetime, timedelta as td
SETTINGS_DIR = os.environ['USH_DIR']
sys.path.insert(0, os.path.abspath(SETTINGS_DIR))
import string_template_substitution
import plot_util
from settings import Paths, ModelSpecs

paths = Paths()
model_colors = ModelSpecs()

def daterange(start, end, td):
   curr = start
   while curr <= end:
      yield curr
      curr+=td

def expand_met_stat_files(met_stat_files, data_dir, output_base_template, RUN_case, 
                          RUN_type, line_type, vx_mask, var_name, model, 
                          eval_period, valid):
    met_stat_files_out = np.concatenate((
        met_stat_files, 
        glob.glob(os.path.join(
            # edit below to define stats archive path. Use '*' as wildcard.
            data_dir, 
            string_template_substitution.do_string_sub(
                output_base_template, 
                RUN_CASE=str(RUN_case), RUN_CASE_UPPER=str(RUN_case).upper(),
                RUN_CASE_LOWER=str(RUN_case).lower(), RUN_TYPE=str(RUN_type), 
                RUN_TYPE_UPPER=str(RUN_type).lower(), 
                RUN_TYPE_LOWER=str(RUN_type).lower(), LINE_TYPE=str(line_type),
                LINE_TYPE_UPPER=str(line_type).upper(), 
                LINE_TYPE_LOWER=str(line_type).lower(),
                VX_MASK=str(vx_mask), VX_MASK_UPPER=str(vx_mask).upper(),
                VX_MASK_LOWER=str(vx_mask).lower(), 
                VAR_NAME=str(var_name), VAR_NAME_UPPER=str(var_name).upper(),
                VAR_NAME_LOWER=str(var_name).lower(), MODEL=str(model), 
                MODEL_UPPER=str(model).upper(), MODEL_LOWER=str(model).lower(),
                EVAL_PERIOD=str(eval_period), 
                EVAL_PERIOD_UPPER=str(eval_period).upper(),
                EVAL_PERIOD_LOWER=str(eval_period).lower(),
                VALID=valid, valid=valid
            )
        ))
    ))
    return met_stat_files_out

def prune_data(data_dir, prune_dir, tmp_dir, output_base_template, valid_range, 
               eval_period, RUN_case, RUN_type, line_type, vx_mask, 
               fcst_var_names, var_name, model_list, interp_pnts):
   print("BEGIN: "+os.path.basename(__file__))
   # Get list of models and loop through
   for model in model_list:
      # Get input and output data
      if model in paths.special_paths:
         if paths.special_paths[model]['data_dir']:
            use_data_dir = paths.special_paths[model]['data_dir']
         else:
            use_data_dir = data_dir
         if paths.special_paths[model]['file_template']:
            use_file_template = paths.special_paths[model]['file_template']
         else:
            use_file_template = output_base_template
      else:
         use_data_dir = data_dir
         use_file_template = output_base_template
      met_stat_files = []
      for valid in daterange(valid_range[0], valid_range[1], td(days=1)):
         met_stat_files = expand_met_stat_files(
            met_stat_files, use_data_dir, use_file_template, RUN_case, RUN_type, 
            line_type, vx_mask, var_name, model,
            #plot_util.get_model_stats_key(model_colors.model_alias, model), 
            eval_period, valid
         )
      pruned_data_dir = os.path.join(
         prune_dir, line_type+'_'+var_name+'_'+vx_mask+'_'+eval_period, tmp_dir
      )
      if not os.path.exists(pruned_data_dir):
         os.makedirs(pruned_data_dir)
      if len(met_stat_files) == 0:
         continue
      with open(met_stat_files[0]) as msf:
         met_header_cols = msf.readline()
      all_grep_output = ''
      if any(interp_pnts):
         print("Pruning "+use_data_dir+" files for model "+model+", vx_mask "
               +vx_mask+", variable "+'/'.join(fcst_var_names)+", line_type "+line_type
               +", interp "+os.environ['INTERP']+", interp points "+'/'.join(interp_pnts))
         filter_cmd = (
            ' | grep " '+vx_mask
            +' " | grep "'+'\|'.join(fcst_var_names)
            +'" | grep " '+line_type
            +' " | grep " '+os.environ['INTERP']
            +' " | grep "'+'\|'.join(interp_pnts)+'"'
         )
      else:
         print("Pruning "+use_data_dir+" files for model "+model+", vx_mask "
               +vx_mask+", variable "+'/'.join(fcst_var_names)+", line_type "+line_type
               +", interp "+os.environ['INTERP'])
         filter_cmd = (
            ' | grep " '+vx_mask
            +' " | grep "'+'\|'.join(fcst_var_names)
            +'" | grep " '+line_type
            +' " | grep " '+os.environ['INTERP']+' "'
         )
      # Prune the MET .stat files and write to new file
      for met_stat_file in met_stat_files:
         ps = subprocess.Popen('grep -R "'
                               +plot_util.get_model_stats_key(model_colors.model_alias, model)
                               +'" '+met_stat_file+filter_cmd,
                               shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding='UTF-8')
         grep_output = ps.communicate()[0]
         all_grep_output = all_grep_output+grep_output
      pruned_met_stat_file = os.path.join(pruned_data_dir,
                                          model+'.stat')
      try:
         with open(pruned_met_stat_file, 'w') as pmsf:
            pmsf.write(met_header_cols+all_grep_output)
      except OSError:
         raise
   print("END: "+os.path.basename(__file__))
