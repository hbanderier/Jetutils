#/bin/bash!

port=16805

srun --nodes=1 --cpus-per-task=2 --mem=200G --time=00:40:00 --qos="job_gratis" bash jupyter_remote_port_forward.sh $port --ServerApp.allow_origin=* --IdentityProvider.token=3b30677a66c7a27927bb717f152b990f18bf8998c9644b1f

