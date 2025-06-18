FROM phusion/passenger-customizable:2.0.0

MAINTAINER David Wilkins <david.wilkins@epfl.ch>


USER root
# Set correct environment variables.
ENV HOME /root


# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# If you're using the 'customizable' variant, you need to explicitly opt-in
# for features. Uncomment the features you want:
#
#   Build system and git.
#   Python support (2.7 and 3.x)
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
    wget \
    python-dev \
    apache2 \
    libapache2-mod-xsendfile \
    libapache2-mod-wsgi \
    apt-utils \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    # Ensure tar and gzip are installed for extraction of data.tar.gz
    && apt-get -y install tar gzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean all

RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py \
    && python2 get-pip.py

# Important because recent versions of pip take correctly care of
# not installing a python2 dependency; I do it for all users
RUN pip2 install --upgrade 'pip==20.3.4'

# set $HOME
ENV HOME /home/app

USER app

RUN pip2 install --upgrade --user ase==3.15.0 matplotlib==2.2.3

RUN mkdir $HOME/code

RUN pip2 install --user llvmlite==0.29.0 numba==0.36.1 tqdm==4.19.5 psutil==5.4.3 future==0.16.0 ipython==5.4 scikit-learn==0.19.1

RUN pip2 install --user matplotlib==2.2.3 Flask==0.12.2 Werkzeug==0.16.0 qe_tools==1.0.0 pymatgen==2017.7.4 celery==4.2.1 redis==2.10.6 monty==1.0.3 kombu==4.2.1 mpmath==1.1.0 cython

USER root

# Copy the webservice code
COPY webservice $HOME/code/webservice

# --- Data inclusion using data.tar.gz ---
# Copy the data archive into the container's /home/app/ directory
COPY data.tar.gz $HOME/

# Create the target directory ($HOME/data) and extract the archive contents into it.
# Assumes data.tar.gz contains a top-level 'data/' directory that you want to strip.
RUN mkdir -p $HOME/data && \
    tar -xzf $HOME/data.tar.gz -C $HOME/data/ --strip-components=1 && \
    rm $HOME/data.tar.gz # Clean up the archive after extraction

# Set proper permissions for the extracted data directory and its contents
RUN chown -R app:app $HOME/data
# --- END Data inclusion ---


# --- SECRET_KEY HARDCODE START ---
# Create the SECRET_KEY file directly in the image with your chosen key
RUN mkdir -p /home/app/code/webservice && \
    echo "your_super_secret_key_1234567890" > /home/app/code/webservice/SECRET_KEY

# Set proper permissions for the SECRET_KEY file
RUN chown app:app /home/app/code/webservice/SECRET_KEY && \
    chmod 600 /home/app/code/webservice/SECRET_KEY
# --- SECRET_KEY HARDCODE END ---

# --- CRITICAL FIX: Ensure 'app' user owns webservice *before* switching to it ---
# This ensures the 'app' user can write the .wsgi file and access the webservice code
RUN chown -R app:app $HOME/code/webservice
# --- END CRITICAL FIX ---


USER app

# Create a proper wsgi file (now writable by 'app' user)
ENV SP_WSGI_FILE=$HOME/code/webservice/seekpath_app.wsgi
RUN echo "import sys" > $SP_WSGI_FILE && \
    echo "sys.path.insert(0, '/home/app/code/webservice')" >> $SP_WSGI_FILE && \
    echo "from seekpath_app import app as application" >> $SP_WSGI_FILE

# Go back to root.
USER root

# Setup apache
ADD ./.docker_files/seekpath-apache.conf /etc/apache2/sites-available/seekpath.conf
RUN a2enmod wsgi && a2enmod xsendfile && \
    a2dissite 000-default && a2ensite seekpath

# Activate apache at startup
RUN mkdir /etc/service/apache
ADD ./.docker_files/apache_run.sh /etc/service/apache/run

# Removed: Reference to create_secret_key.sh as it's now hardcoded in Dockerfile
# # Set startup script to create the secret key
# RUN mkdir -p /etc/my_init.d
# ADD ./.docker_files/create_secret_key.sh /etc/my_init.d/create_secret_key.sh

# Web
EXPOSE 80

RUN cd $HOME/code/webservice && python setup_cython.py build_ext --inplace && cd -

# Final cleanup, in case it's needed
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*