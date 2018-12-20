OTHERFILE=FaBP.m \
	README \
   	Makefile \
      	license.txt

FABPFILE=${OTHERFILE} \
	DATA/A.mat \
     	DATA/pseudopriors.mat \

demo: 
	matlab -r FaBP

zip: tar 

tar: ${FABPFILE}
	zip fabp.zip ${FABPFILE}
