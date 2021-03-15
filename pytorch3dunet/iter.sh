conda activate acunet
for p in ./io/queue/* ; do
  echo ${p:11}
  mv $p ./io/test/${p:11}
  python predict.py --config ../resources/test_config_ac.yaml
  mv ./io/test/${p:11} $p
  # python denoise.py $p ./result/${p:7}
done
conda deactivate
