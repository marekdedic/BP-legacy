#!/bin/sh

RES="../BP/avclass_results.txt"

cd ../avclass
rm -f $RES
for f in $(find ../threatGridSamples2 -name '*.vt.json'); do
	echo -n $f >> $RES
	echo -n "	" >> $RES
	./avclass_labeler.py -vt $f 2>/dev/null >> $RES
done;

