#!/usr/bin/bash

input_file=$1
rst_file=$2



cur_dir=`dirname $0`
#diagnosis_file=${input_file}_f1
#cut_words_file=${input_file}_cut
#echo $cut_words_file

start=$(date +%s)  

#printf "cut diagnosis...\n"
#cut -f 1 $input_file > $diagnosis_file


#printf "cut sentence to words...\n"
#nohup python  cut_words.py $diagnosis_file $cut_words_file > $cur_dir/log/cut_words.py.log 2>&1  &



printf "extract features and clustering...\n"
python gen_tf-idf.py $input_file $rst_file > $cur_dir/log/gen_tf-idf.py.log 2>&1

printf "sort by cluster label...\n"
sort -k 1 $rst_file > ${rst_file}_sorted

end=$(date +%s)  
time=$(( $end - $start ))  
printf "total time cost $time \n"
