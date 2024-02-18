#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="/home/george/unbacked_up_data/sdss_quasar_dps/*"
for f in $FILES
do
    nfiles=`ls $f | wc -l`

    if [ $nfiles -lt 3 ]
    then
	echo $(basename $f) $nfiles
    fi
done
