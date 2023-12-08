cd ..
models=("None" "hessian" "sgd")
attacks=("deep_fool" "pgd" "fsgm" "universal" "boundary")
numsamples=3000
t=100

for mod in "${models[@]}"
do
    for att in "${attacks[@]}"
    do
        echo "Model: $mod"
        echo "Attack: $att"
        echo "Number of samples: $numsamples"
        echo "T steps: $t"
        python eval_attacks.py --attack_name $att --unlearning $mod --num_samples $numsamples --t $t
    done
done