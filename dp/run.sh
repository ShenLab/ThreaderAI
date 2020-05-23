
base_dir=/data/hz2529/zion/casp/CASP13/results_theraderai/2020-01-26-15-18-47-epoch14
base_dir=/data/hz2529/zion/casp/CASP13/results_theraderai/2020-02-07-19-51-01-epoch15
base_dir=/data/hz2529/zion/casp/CASP13/results_theraderai_bc90/2020-02-07-19-51-01-epoch15
if [ $# -ne 1 ]
then
	echo "usage:<1> name"
	exit
fi

name=${1}
query_feature_dir=/data/hz2529/zion/casp/CASP13/sequence_new
template_feature_dir=/data/hz2529/zion/pdbdata/pool/sequence_bc70_2018-01-01_uniclust30_2017_10
template_feature_dir=/data/hz2529/zion/pdbdata/pool/structure
ls ${base_dir}/${name}/score/ | cut -f 1 -d '.' > ${base_dir}/tm.list

tmlist=/data/hz2529/zion/pdbdata/pool/bc90.list
tmlist=${base_dir}/tm.list

./threading --template_list $tmlist \
	--score_dir ${base_dir}/${name}/score \
	--output ${base_dir}/${name}/${name}.aligned_pairs \
	--logtostderr \
	--alpha 0.3



#template_list=./t.list
#query_feature_dir=/data/hz2529/zion/casp/CASP13/sequence_uniclust30_2016_09
#template_feature_dir=/data/hz2529/zion/pdbdata/copy2


output=${base_dir}/${name}/${name}.threaderai
 
python3.7 ./show_alignment.py \
	--aligned_pairs ${base_dir}/${name}/${name}.aligned_pairs \
	--query ${name} \
	--sequence_dir ${query_feature_dir} \
	--structure_dir ${template_feature_dir} \
	--output ${output}

#build_model
bash /home/hz2529/repos/structure_utility/casp/calc_threaderai_model.sh ${name}
