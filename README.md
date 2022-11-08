# RecommendationSystem

## Getting started

To use this project, please clone the repository to your local system. </br>
Also, make sure to install docker, al it will make runnig it easier. </br>

## Running using docker

1- cd into cloned repo
2- run the following command: docker build --tag RecommendationSystem. 
3- run the following command: docker run -it -p 8999:80 RecommendationSystem

## API endpoints

The API currently supports the following features: </br>
1- /hello: test the api. </br>
2- /predict[user info]: predicts the recommended contract based on the user info. </br>

## Path Modifications 
If you are using this project on windows change the path in src/modeling.py in line 59 58
