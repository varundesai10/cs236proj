cd ..
pwd
NUM_SAMPLES=1000
DATASET="cifar"
TARGET="dog"
python run_generate_attacks.py -n $NUM_SAMPLES -d $DATASET -t $TARGET