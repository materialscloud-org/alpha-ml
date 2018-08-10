# Use phusion/baseimage as base image. To make your builds
# reproducible, make sure you lock down to a specific version, not
# to `latest`! See
# https://github.com/phusion/baseimage-docker/blob/master/Changelog.md
# for a list of version numbers.
# Note also that we use phusion because, as explained on the
# http://phusion.github.io/baseimage-docker/ page, it automatically
# contains and starts all needed services (like logging), it
# takes care of sending around signals when stopped, etc.
##
# Actually, I use passenger-full that already has python
# https://github.com/phusion/passenger-docker#using
FROM phusion/passenger-customizable:0.9.19

MAINTAINER Felix Musil <felix.musil@epfl.ch>


USER root
# Set correct environment variables.
ENV HOME /root


# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# If you're using the 'customizable' variant, you need to explicitly opt-in
# for features. Uncomment the features you want:
#
    #   Build system and git.
    #   Python support (2.7 and 3.x - it is 3.5.x in this ubuntu 16.04)
RUN /pd_build/utilities.sh && \
    /pd_build/python.sh

##########################################
############ Installation Setup ##########
##########################################

# Install required software

# First, install pip (for python 2)
# Install also Apache (nginx doesn't have the X-Sendfile support
# that we use)
## Note: to install instead pip3 for python3, install the package python3-pip
## However, then one has to configure the web server to use wsgi with python3
RUN apt-get update \
    && apt-get -y install \
    apt-utils \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean all

RUN apt-get update \
    && apt-get -y remove \
		python python2.7 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean all

RUN apt-get update \
    && apt-get -y install \
    python python2.7  \
    python-pip \
    apache2 \
    libapache2-mod-xsendfile \
    libapache2-mod-wsgi \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean all

# set $HOME
ENV HOME /home/app


USER app
### INSTALL QUIP
RUN pip2 install numpy==1.13 scipy==1.0.0 Flask==0.12.2 qe_tools==1.0.0 pymatgen==2017.7.4
RUN pip2 install --upgrade --user ase==3.15.0

# fetch QUIP, GAP and shortcut files to install linux_x86_64_gfortran architecture
RUN mkdir $HOME/git \
    && cd $HOME/git/ \
    && git clone --recursive https://github.com/libAtoms/QUIP.git \
    && cd $HOME/git/QUIP \
    && git checkout 95da7aad1f48a6f78e5effce8de82f114e37606f \
    && mkdir $HOME/git/QUIP/build/ \
    && mkdir $HOME/git/QUIP/build/linux_x86_64_gfortran/

COPY GAP $HOME/git/QUIP/src/GAP
COPY Makefile.inc $HOME/git/QUIP/build/linux_x86_64_gfortran/

# Install QUIP and quippy for the user
RUN cd $HOME/git/QUIP/ \
    && export QUIP_ARCH=linux_x86_64_gfortran \
    && make

RUN cd $HOME/git/QUIP/ \
    && export QUIP_ARCH=linux_x86_64_gfortran \
    && make quippy \
    && make QUIPPY_INSTALL_OPTS=--user install-quippy

RUN mkdir $HOME/code

RUN pip2 install numba==0.36.1 tqdm==4.19.5 psutil==5.4.3 future ipython==5.4 scikit-learn==0.19.1 celery==4.1.0 redis==2.10.6

RUN cd $HOME/git/  \
    && git clone https://github.com/cosmo-epfl/glosim2.git


USER root

# Actually, don't download, but get the code directly from this repo
#COPY predictor.py $HOME/code/predictor.py
COPY webservice $HOME/code/webservice
RUN mkdir $HOME/data
# COPY ../data $HOME/data
# COPY data $HOME/data
#COPY ./README.rst README.rst
#COPY ./MANIFEST.in MANIFEST.in
#COPY ./LICENSE.txt LICENSE.txt
#COPY run_tests.py $HOME/code/run_tests.py

# Set proper permissions
RUN chown -R app:app $HOME/code/webservice
RUN chown -R app:app $HOME/data
#RUN chown app:app $HOME/code/predictor.py $HOME/code/run_tests.py

USER app

# Create a proper wsgi file
#
ENV SP_WSGI_FILE=$HOME/code/webservice/seekpath_app.wsgi
RUN echo "import sys" > $SP_WSGI_FILE && \
    echo "sys.path.insert(0, '/home/app/code/webservice')" >> $SP_WSGI_FILE && \
    echo "from seekpath_app import app as application" >> $SP_WSGI_FILE

# Go back to root.
# Also, it should remain as user root for startup
USER root

# Setup apache
# Disable default apache site, enable seekpath site; also
# enable needed modules
ADD ./.docker_files/seekpath-apache.conf /etc/apache2/sites-available/seekpath.conf
RUN a2enmod wsgi && a2enmod xsendfile && \
    a2dissite 000-default && a2ensite seekpath

# Activate apache at startup
RUN mkdir /etc/service/apache
ADD ./.docker_files/apache_run.sh /etc/service/apache/run

# Set startup script to create the secret key
RUN mkdir -p /etc/my_init.d
ADD ./.docker_files/create_secret_key.sh /etc/my_init.d/create_secret_key.sh

# Web
EXPOSE 80

# Final cleanup, in case it's needed
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
