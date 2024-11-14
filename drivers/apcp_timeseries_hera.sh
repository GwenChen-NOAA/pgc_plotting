#!/bin/bash
#SBATCH --account=ovp
#SBATCH --job-name=htar_cam
#SBATCH --output=/scratch2/NCEPDEV/ovp/Marcel.Caron/test/out/pgc_plotting.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --partition=hera

set -x

HOMEpgc_plotting="/scratch2/NCEPDEV/ovp/Marcel.Caron/dev/pgc_plotting"
machine="hera"
source ${HOMEpgc_plotting}/hera.env

HOMEwcoss2="/lfs/h2/emc/vpppg/noscrub/marcel.caron"
HOMEhera="/scratch2/NCEPDEV/ovp/Marcel.Caron/dev"
machine="hera"
export COMPONENT="global_ens"
if [ "${machine}" = "hera" ]; then
   HOME=${HOMEhera}
   export STAT_OUTPUT_BASE_DIR="/scratch2/NCEPDEV/stmp1/Marcel.Caron/evs/v2.0/stats/${COMPONENT}"
elif [ "${machine}" = "wcoss2" ]; then
   HOME=${HOMEwcoss2}
   export STAT_OUTPUT_BASE_DIR="/lfs/h2/emc/vpppg/noscrub/emc.vpppg/gefs_pgc_stats/evs/v2.0/stats/${COMPONENT}"
else
   echo "Unknown machine: ${machine}"
   exit 1
fi
${HOMEpgc_plotting}/parm/apcp_timeseries.config


set +x
