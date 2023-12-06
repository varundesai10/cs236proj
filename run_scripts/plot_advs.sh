cd ../attacks_adv
attacks=("deep_fool" "pgd" "fsgm" "universal" "boundary")



for att in "${attacks[@]}"
do
    echo "Attack: $att"
    python plot_diffused.py -a $att
done
