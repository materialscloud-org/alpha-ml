# NMR_predictor_app

To build the docker image:

sudo docker build -t nmr_app .

To access the image's CL with the test data:

 sudo docker run -it --rm -v /local/git/NMR_predictor_app/data:/home/app/data nmr_app /bin/bash

sudo docker run -it --rm -v /Users/iMac/PycharmProjects/nmr_test/data:/home/app/data nmr_app 

sudo docker run -p 5050:80 --rm -v /Users/iMac/PycharmProjects/nmr_test/data:/home/app/data  test 