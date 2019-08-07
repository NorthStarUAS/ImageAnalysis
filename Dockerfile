# Builds an environment that runs the 

FROM continuumio/anaconda3:latest
LABEL Name=directgeoref Version=0.0.1
EXPOSE 3000

ENV PATH /opt/conda/bin/
SHELL ["/bin/bash", "-c"]

# Define the working dir where to put all the source code
WORKDIR /app
# And the location of where to link the images when mounting the volume that holds the images, camera JSON and 
VOLUME [ "/data" ]

# Update conda if required
RUN conda update conda

# Install all the required packages
RUN conda install -c conda-forge opencv && \
pip install navpy piexif geojson panda3d opencv-contrib-python pylint rope matplotlib

# Download the required props package
ADD https://github.com/AuraUAS/aura-props/archive/master.zip /app/

# extract the compressed files and install
RUN python -m zipfile -e /app/master.zip /app/ && \
cd aura-props-master/python && \
python setup.py install 

# Add the source code last to ensure that the rebuild is as quick as possible
ADD . /app

ENTRYPOINT ["/bin/bash", "-c"]
CMD cd /app

# To run, use the following commands:
# docker run -d --name imageanalysis -v location/on/your/computer:/data imageanalysis
# docker exec -it imageanalysis /bin/bash