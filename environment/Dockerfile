# hash:sha256:adcb55f3fab4f51f72b3943f1ec952b9af00187729a537d12dbf0b84775b398b
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=21.2.6-0ubuntu0.1~20.04.2 \
        libglib2.0-0=2.64.6-1~ubuntu20.04.8 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyter==1.0.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    imutils==0.5.4 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    opencv-python==4.10.0.84 \
    pandas==2.2.3 \
    plotly==5.24.1 \
    pyproj==3.6.1 \
    scikit-image==0.24.0 \
    scikit-learn==1.6.1 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    tilemapbase==0.4.7
