# CRAG-MLAdversary

## Project Overview
This project focuses on Machine Learning Adversary research. We aim to develop various adversarial attacks against an image classifier to drastically reduce its performance and subsequently enhance its adversarial robustness.

## Team Information
- **Team Name:** CRAG-MLAdversary

## Team Members
- **Chibuike E. Ugwu**
- **Guo, Jiawei**
- **Azza Fadhel**
- **Stirewalt, Tashi Russell**


## Usage information
- Make sure Node.js is locally installed. Download available at https://nodejs.org/en/download
- Execute node -v in terminal to make sure it downloaded and pathed properly
- Naviagte to frontend/src/api
- Use ```npm install``` -> ```npm run serve``` to launch front end
- Run .sql to create a database

## To make a prediction with the classifier, run on the terminal as follows:
**Dependencies:**
- !pip install torch
- !pip install torchvision
- can generally be run in a conda environment

**How to run:** 
- !python3 prediction.py "url-of-image-1" "url-of-image-1" ...
- Below is an example:
- !python3 prediction.py "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg" "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Rottweiler_standing_facing_left.jpg/800px-Rottweiler_standing_facing_left.jpg"
- **Output can be found in the "output.json" file**
