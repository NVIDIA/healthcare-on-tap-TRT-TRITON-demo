#!/usr/bin/env bash

#Enable NV Dashboard
# jupyter labextension install jupyterlab-nvdashboard

netron & jupyter lab ./ --ip 0.0.0.0 --port 8888 --allow-root --no-browser
