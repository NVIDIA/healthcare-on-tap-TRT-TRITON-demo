FROM nvcr.io/nvidian/pytorch:20.11-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    nodejs



# Set up IPYwidgets
RUN pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension

# Install Netron
RUN pip install netron

# Install TRITON Clients
RUN pip install nvidia-pyindex
RUN pip install tritonclient[all]

# Install MONAI
RUN pip install monai[all]

# Install PYCUDA
RUN pip install pycuda

# Install/Enable GPU dashboards
RUN pip install jupyterlab-nvdashboard 
RUN PATH="/usr/local/nvm/versions/node/v15.0.1/bin:$PATH" jupyter labextension install jupyterlab-nvdashboard
