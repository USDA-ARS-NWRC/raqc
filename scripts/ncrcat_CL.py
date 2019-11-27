import sys, os
from subprocess import check_output
import argparse
import copy
from raqc.utilities import format_date
import pandas as pd

'''used to agglomerate awsm_daily run outputs into one .nc file'''

fp = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description = 'agglomerate single date netCDF \
                                                files into single file')
parser.add_argument('--dir-in', default = os.getcwd(),
                    help = 'directory where awsm_daily run folders reside')
parser.add_argument('--base-file', default = 'snow.nc', help = 'filename in \
                    subdirectories being concatenated i.e. snow.nc')
parser.add_argument('--file-out', required = True, help = 'filename and path of \
                    new file with file type extension i.e. .nc')
parser.add_argument('--start-date', help = 'start date in format YYYYMMDD',
                        type = int)
parser.add_argument('--end-date', help = 'end date in format YYYYMMDD', \
                    type = int)
parser.add_argument('--increment', type = int,
                    help = 'ex) if there are n = 100 subdirectories, --increment=2 \
                            will result in days 1,3,5...99 being concatenated \
                            into one .nc file with ncrcat \
                            If --increment is not specified, then daily will be \
                            imputed.  In this case 100 days will be concatenated \
                            into one .nc file')

args = parser.parse_args()
dir_in = os.path.abspath(os.path.expanduser(args.dir_in))

# 1) Get list of full paths to subdirectories
subD_paths = [os.path.join(dir_in, subD) for subD in os.listdir(dir_in)]
# ensure that only directories are included
is_dir = [os.path.isdir(subD) for subD in subD_paths]
subD = [subD_paths[id] for id, is_dir in enumerate(is_dir) if is_dir]
subD.sort()

files_in = []
for sub in subD:
    files_in.append(os.path.join(sub, args.base_file))


# 2) get datetime object using short class method from utilities
# get first and last date of subdirectory in --dir-in arg (main directory)
subdirectory_start, subdirectory_end = subD[0], subD[-1]
subdirectory_start = subdirectory_start.split('/')[-1][3:]
subdirectory_end = subdirectory_end.split('/')[-1][3:]
# Turn string into datetime object
subdirectory_start = format_date(subdirectory_start)
subdirectory_end = format_date(subdirectory_end)

# 3) Ensure dates passed within bounds
# check if user passed args to select specified dates using start, stop, and
# increment. If not, default to main directory date bounds and increment = Days
try:
    subset_start = format_date(args.start_date)
except TypeError:
    subset_start = subdirectory_start
try:
    subset_end = format_date(args.end_date)
except TypeError:
    subset_end = subdirectory_end
if args.increment != None:
    increment = pd.to_timedelta(args.increment, unit = 'D')
else:
    increment = pd.to_timedelta(1, unit = 'D')

# ensure start and stop dates are within subdirectory date bounds
if increment > subset_end - subset_start:
    sys.exit('increment is greater than number of days in timespan\n'
                '{0} exiting {1}').format('-'*30, '-'*30)

if subset_start < subdirectory_start:
    sys.exit('the start date is earlier than first available .nc file \n'
            '{0} exiting {1}').format('-'*30, '-'*30)

if subset_end > subdirectory_end:
    sys.exit('the end date is later than last available .nc file \n'
            '{0} exiting {1}').format('-'*30, '-'*30)

# 4) Get list of filepaths in directory to pass in argument
timespan_subdirectory = subdirectory_end - subdirectory_start
timespan_subset = subset_end - subset_start
# create list of num days to add to start date for producing datetime objects
# of all dates in directory
temp_list = list(range(0, timespan_subdirectory.days))
subdirectory_dates = [subdirectory_start + pd.to_timedelta(d, unit = 'D') for d in temp_list]
# create list of num days to add to create list of datetime objects for specified
# start, stop and increment
temp_list = list(range(0, timespan_subset.days + 1, increment.days))
subset_dates = [subset_start + pd.to_timedelta(d, unit = 'D') for d in temp_list]

# get indices of subdirectory dates requested in subset
ids = []
for ids1, subD_date in enumerate(subdirectory_dates):
    for subS_date in subset_dates:
        if subS_date == subD_date:
            ids.append(ids1)
            break

files_in_temp = files_in.copy()
files_in = [files_in_temp[ids] for ids in ids]
files_in = ' '.join(files_in)

file_out = os.path.abspath(os.path.expanduser(args.file_out))
# NOTE: cmd is command line call.  IF script needs to be modified to concatenate
# a different band(variable), simply replace 'thickness' with desired band name.

cmd = 'ncrcat -v {} {} {}'.format('cold_content', files_in, file_out)

check_output(cmd, shell=True)  # t = text output
