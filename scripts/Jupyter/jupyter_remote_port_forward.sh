#!/bin/env bash
# UniBE 2020
# this tool provides portforwarding from the submitted node to the compute node and
# starts jupyter lab
#  mandes.schoenherr@id.unibe.ch

port=${1:-15051}

echo "Jupyter lab will be started on $(hostname) and establish port farwarding from $SLURM_SUBMIT_HOST on port $port"

ssh -N -R ${port}:localhost:${port} $SLURM_SUBMIT_HOST &

XDG_RUNTIME_DIR="" uv run --with jupyter jupyter lab --ServerApp.ip='0.0.0.0' --port ${port} --no-browser ${@:2}

