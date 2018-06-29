# Shift-ML app

To build the docker image:

sudo docker build -t nmr_app .

To access the image's CL with the test data:

sudo docker run -p 5050:80 --rm -v /Users/albert/PycharmProjects/data:/home/app/data  nmr_app 
 

 The data model_sp_*.pck needs to be in mounted local directory so it can be loaded by the app from /home/app/data
 
# Docker compose`
 
 To build and run the docker images
 ``` 
 docker-compose -f docker-compose_albert.yml up --build
 docker-compose -f docker-compose_felix.yml up --build
  ```
 OR (without stdout redirection)
 ```
 docker-compose -f docker-compose_felix.yml build
 docker-compose -f docker-compose_felix.yml start
 docker-compose -f docker-compose_felix.yml stop
 ```
 and to clean up the images
 ```
 docker-compose rm -f
 ```
 # To run the app locally
 In a terminal in websevice run:
 ``` 
 launchctl load /usr/local/opt/redis/homebrew.mxcl.redis.plist  OR 
 celery worker -l info -A seekpath_app.celery
 ```
 And in another in webservice:
 ``` 
 python seekpath_app.py
 ```
 
 # Additional packages
 
```
pip install celery
```
 
# To use redis

redis is a framework to simplify the use of database/broker in the flask app.

* Install
```
 brew install redis
 pip install redis
 ```
 Or
 ```
 sudo apt-get install redis-server
 pip install redis
 ```
 
 * Run in background
 ```
 launchctl load /usr/local/opt/redis/homebrew.mxcl.redis.plist
 ```
 Linux should already be running is in the background.

#to run locally
launchctl load /usr/local/opt/redis/homebrew.mxcl.redis.plist    #launch redis in the background
celery worker -l info -A seekpath_app.celery #in terminal 1 in /webservice
python seekpath_app.py #in terminal 2 in /websercive
