n_nodes=$1
echo "Killing $n_nodes nodes"
rank=0
for i in $(cat computers_name | head -n $n_nodes)
do
    ssh -oStrictHostKeyChecking=no $i "tmux kill-session -t birdrec" &
    echo Killed node $i with rank $rank
    ((rank++))
done
