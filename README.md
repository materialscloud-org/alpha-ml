# Shift-ML app

To build the docker image:

sudo docker build -t nmr_app .

To access the image's CL with the test data:

sudo docker run -p 5050:80 --rm -v /Users/iMac/PycharmProjects/nmr_test/data:/home/app/data  nmr_app 
 
 The data model_sp_*.pck needs to be in mounted local directory so it can be loaded by the app from /home/app/data