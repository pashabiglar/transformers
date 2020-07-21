#!/bin/bash

: <<COMMENTBLOCK
Authors: Sara Willis, Tom Hicks, Dianne Patterson
Date: March 6, 2020
Purpose: To create a qsub array of jobs to process multiple subjects.
This is more efficient than a bash for-loop and especially suitable for 
processing BIDS format data.

Each subject-job is spawned separately so that:
-Crashing one does not crash the others;
-You do not have to estimate how long multiple jobs require;
-Each of your jobs will be inserted in the queue when time is available.
Input: path to script file and path to subject list:
e.g., qsubr Scripts/echo2.sh SubjectLists/s.txt
COMMENTBLOCK


if [ $# -lt 1 ]; then
  echo "Usage: script [subjects-file]"
  echo "where:"
  echo "   script is the script to be submitted"
  echo "   subjects-file contains subject number, one per line (default './subjects.txt')"
  echo "   Make sure there is a final hard return after the last entry"
  echo ""
  exit 1
fi

# Export the name of the subject file. This is necessary if the subject files are in a directory somewhere
# Ensure that there is a hard return after the last entry in the subject file.
# If no subject list is specified, default to subjects.txt in the current working directory.
export SUBJS=${2:-subjects.txt}
# Count the number of lines in the subject file, so we know how many entries will be in the qsub array.
NSUBJS=`wc -l <${SUBJS}`
# Get the basename of the script without the file extension, so we can use it to name the output job
SCRIPT=`basename $1 | sed 's/\.[^.]*$//'`
# Get the basename of the subject list without the file extension so we can also use it to name the output job
LIST_FILE=`basename ${SUBJS} | sed 's/\.[^.]*$//'`
# Create the jobname
JOBNAME=${SCRIPT}_${LIST_FILE}
# Call qsub for each subject from the list, with the jobname we created 
# and using the -J range flag to define the number of times to loop.
# Finally, add the name of the script to run with qsubr
qsub -v"SUBJS" -N ${JOBNAME} -J "1-${NSUBJS}" ${1:-echo2.sh}