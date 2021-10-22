m="bone"

conda activate acunet
while read q; do
  echo "$q"
  OLD_IFS="$IFS"
  IFS=","
  qarray=($q)
  IFS="$OLD_IFS"
  p=${qarray[0]}
  c=${qarray[1]}
  
  mv ../datasets/$m/queue/$p ../datasets/$m/test/$p
  python predict_crop.py --config ../resources/test_config_$m.yaml <<< $c
  mv ../datasets/$m/test/$p ../datasets/$m/queue/$p
done < /home/shkim/Libraries/pytorch-3dunet/datasets/$m/config_test
conda deactivate
