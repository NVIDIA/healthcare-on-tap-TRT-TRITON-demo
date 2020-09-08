FROM nvcr.io/nvidian/pytorch:20.06-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    nodejs



# Set up IPYwidgets
RUN pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension

# Install Netron
RUN pip install netron

# Install TRITON Clients
ARG TRITON_CLIENTS_URL=https://github.com/NVIDIA/triton-inference-server/releases/download/v1.14.0/v1.14.0_ubuntu1804.clients.tar.gz
RUN mkdir -p /opt/nvidia/triton-clients \
    && curl -L ${TRITON_CLIENTS_URL} | tar xvz -C /opt/nvidia/triton-clients
RUN pip install --no-cache-dir --upgrade setuptools wheel \
    && pip install --no-cache-dir /opt/nvidia/triton-clients/python/*.whl

# Install GPU dashboards

RUN pip install jupyterlab-nvdashboard
# jupyter labextension install jupyterlab-nvdashboard
