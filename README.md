# NMR_predictor_app

To build the docker image:

sudo docker build -t nmr_app .

To access the image's CL with the test data:

 sudo docker run -it --rm -v /local/git/NMR_predictor_app/data:/home/app/data nmr_app /bin/bash
