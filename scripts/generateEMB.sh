#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-00:15:00     # 1 day and 15 minutes
#SBATCH --output=my.stdout
#SBATCH --mail-user=useremail@address.com
#SBATCH --mail-type=ALL
#SBATCH --job-name="emb"
#SBATCH -p intel # This is the default partition, you can use any of the following; intel, batch, highmem, gpu

begin=$(date +%s)

function generate_emb()
{
	echo "Generating file " $1/$2
	python embedding/preprocessemb.py $1/$2
}


dir=out/idafiles
home=$PWD
cd ..
echo "Generating idafiles in " $dir
cd $dir
declare -a array=(`ls`)
cd $home
cd ..
max=${#array[@]}
echo "containing " $max "files"

rsnum=10
times=$(expr $max / $rsnum)
PID=()
for((i=1; i<=max; )); do
	echo "job i = " $i
	for((Ijob=0; Ijob<rsnum; Ijob++)); do
		if [[ $i -gt $max ]]; then
			break;
		fi
		if [[ ! "${PID[Ijob]}" ]] || ! kill -0 ${PID[Ijob]} 2> /dev/null; then
			echo "Ijob = " $Ijob
			j=$((i-1))
			filedir=${array[$j]}
			generate_emb $dir $filedir &
			PID[Ijob]=$!
			i=$((i+1))
		fi
	done
	sleep 1
done
wait
#python codesearch/embedding/preprocessemb.py bigdata/binaryssl

end=$(date +%s)
spend=$(expr $end - $begin)
echo "spend time" $spend "seconds"
