#/bin/bash!
./jupyter-compute 16805 --jupyter_args="--ServerApp.allow_origin=* --IdentityProvider.token=3b30677a66c7a27927bb717f152b990f18bf8998c9644b1f" --nodes=1 --cpus-per-task=6 --mem=100G --time=02:30:00 --qos="job_gratis" $@
