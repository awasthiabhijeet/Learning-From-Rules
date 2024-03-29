#!/usr/bin/env bash
logdir=logs/MITR
mkdir -p $logdir # logs are dumped here

# Keep any one group of flags (4 consecutive lines) active at any time and run the corresponding experiment

# USE THIS FOR IMPLY LOSS
# declare -a arr=("implication") # ImplyLoss (Our method) in Table 2 Column3 (MITR) 
# declare -a gamma_arr=(0.1)
# declare -a lamda_arr=(0.1) # not actually used 
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10) # (10 independent models were trained and numbers obtained were averaged)

# USE THIS FOR POSTERIOR REG.
# declare -a arr=("pr_loss") # Posterior Reg. in Table 2 Column3 (MITR) 
# declare -a gamma_arr=(0.01)
# declare -a lamda_arr=(0.1) # not actually used
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10) # 

# USE THIS FOR L+Usnorkel
# declare -a arr=("label_snorkel") # L+Usnorkel in Table 2 Column3 (MITR)
# declare -a gamma_arr=(0.05)
# declare -a lamda_arr=(0.1) # not actually used
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

# USE THIS FOR Nosie-Tolerant
# declare -a arr=("gcross") 
# declare -a gamma_arr=(0.01)
# declare -a lamda_arr=(0.6) 
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

# USE THIS FOR L+Umaj
# declare -a arr=("gcross") 
# declare -a gamma_arr=(0.01)
# declare -a lamda_arr=(0)
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

# USE THIS Snorkel-Noise-Tolerant
# declare -a arr=("gcross_snorkel")
# declare -a gamma_arr=(0.001)
# declare -a lamda_arr=(0.6) 
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

# # USE THIS FOR L2R
# declare -a arr=("learn2reweight") # L2R in Table 2 Column3 (MITR)
# declare -a gamma_arr=(0.1) # not actually used
# declare -a lamda_arr=(0.0001) # meta-learning rate
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

#USE THIS FOR Only-L
# declare -a arr=("f_d") # Only-L in Table 2 Column3 (MITR) 
# declare -a gamma_arr=(0.1) # not actully used
# declare -a lamda_arr=(0.1) # not actully used
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)


EPOCHS=100
LR=0.0003
CKPT_LOAD_MODE=mru
DROPOUT_KEEP_PROB=0.8
VALID_PICKLE_NAME=validation_processed.p
U_pickle_name="U_processed.p"
D_PICKLE_NAME="d_processed.p"
USE_JOINT_f_w=False


for MODE in "${arr[@]}"
do
   echo "$MODE"
   mode=$MODE
   for GAMMA in "${gamma_arr[@]}"
   do
      for LAMDA in "${lamda_arr[@]}"
      do
         for Q in "${model_id[@]}"
         do
            nohup ./MITR.sh "$MODE"_"$GAMMA"_"$LAMDA"_"$Q" $mode $EPOCHS $LR $CKPT_LOAD_MODE \
            $DROPOUT_KEEP_PROB $D_PICKLE_NAME $VALID_PICKLE_NAME \
            $U_pickle_name $GAMMA $LAMDA $USE_JOINT_f_w > $logdir/"$MODE"_"$GAMMA"_"$LAMDA"_"$Q".txt &
         done
      done
   done  
done
