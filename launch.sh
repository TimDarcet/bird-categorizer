n_nodes=$1
echo $n_nodes

for i in $(cat computers_name | head -n $n_nodes)
do
        # ssh -oStrictHostKeyChecking=no $i "tmux new-session -d -s birdrec \"cd ~/bird-recognition-mva/bird-categorizer && conda activate pytorch && python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$n_nodes --node_rank=1 --master_addr=\\\"129.104.254.39\\\" --master_port=1234 ./distrib_train.py\"" &
        echo $c
done
