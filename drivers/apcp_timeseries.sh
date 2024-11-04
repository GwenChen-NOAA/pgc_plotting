#PBS -S /bin/bash
#PBS -N pgc_precip_timeseries
#PBS -j oe
#PBS -S /bin/bash
#PBS -q dev
#PBS -A VERF-DEV
#PBS -l walltime=01:00:00
#PBS -l place=vscatter:exclhost,select=1:ncpus=1:ompthreads=1:mem=128GB
#PBS -l debug=true

set -x

HOMEpgc_plotting="/lfs/h2/emc/vpppg/noscrub/marcel.caron/pgc_plotting"
source ${HOMEpgc_plotting}/run.env
${HOMEpgc_plotting}/sample_configs/apcp_timeseries.config

set +x
