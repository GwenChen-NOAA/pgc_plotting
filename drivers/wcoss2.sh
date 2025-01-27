#PBS -S /bin/bash
#PBS -N pgc_precip_wcoss2
#PBS -j oe
#PBS -S /bin/bash
#PBS -q dev
#PBS -A VERF-DEV
#PBS -l walltime=06:00:00
#PBS -l place=vscatter:exclhost,select=1:ncpus=1:ompthreads=1:mem=128GB
#PBS -l debug=true

set -x

HOMEpgc_plotting="/lfs/h2/emc/vpppg/noscrub/Lichuan.Chen/pgc_plotting"
machine="wcoss2"
source ${HOMEpgc_plotting}/wcoss2.env

HOMEwcoss2="/lfs/h2/emc/vpppg/noscrub/Lichuan.Chen"
HOMEhera="/scratch2/NCEPDEV/ovp/Lichuan.Chen"
#machine="wcoss2"
export COMPONENT="global_ens"
if [ "${machine}" = "hera" ]; then
   HOME=${HOMEhera}
   export STAT_OUTPUT_BASE_DIR="/scratch2/NCEPDEV/stmp1/Lichuan.Chen/evs/v2.0/stats/${COMPONENT}"
elif [ "${machine}" = "wcoss2" ]; then
   HOME=${HOMEwcoss2}
   export STAT_OUTPUT_BASE_DIR="/lfs/h2/emc/vpppg/noscrub/emc.vpppg/gefs_pgc_stats/evs/v2.0/stats/${COMPONENT}"
else
   echo "Unknown machine: ${machine}"
   exit 1
fi
field="${field:-apcp24}"
metric="${metric:-bss_1mm}"
plottype="${plottype:-timeseries}"
${HOMEpgc_plotting}/parm/${field}_${metric}_${plottype}.config

set +x
