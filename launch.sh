n_nodes=$1
echo "Launching $n_nodes nodes"
rank=0
for i in $(cat computers_name | head -n $n_nodes)
do
    ssh -oStrictHostKeyChecking=no $i "tmux new-session -d -s birdrec \"cd ~/bird-recognition-mva/bird-categorizer && conda activate pytorch && python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$n_nodes --node_rank=$rank --master_addr=\\\"129.104.254.39\\\" --master_port=1234 ./distrib_train.py\"" &
    echo Launched node $i with rank $rank
    ((rank++))
done
