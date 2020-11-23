n_nodes=$1
echo "Launching $n_nodes nodes"
rank=0
mkdir -p logs/stdout
mkdir -p logs/stderr
for i in $(cat computers_name | head -n $n_nodes)
do
    ssh -oStrictHostKeyChecking=no $i "tmux new-session -d -s birdrec \"source ~/.bashrc && cd ~/bird-recognition-mva/bird-categorizer && conda activate pytorch && python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$n_nodes --node_rank=$rank --master_addr=\\\"anchois.polytechnique.fr\\\" --master_port=1234 ./distrib_train.py >logs/stdout/log.$i 2>logs/stderr/log.$i\"" &
    echo Launched node $i with rank $rank
    ((rank++))
done
