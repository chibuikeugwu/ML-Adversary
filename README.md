# CRAG-MLAdversary

## Project Overview
This project focuses on Machine Learning Adversary research. We aim to develop various adversarial attacks against an image classifier to drastically reduce its performance and subsequently enhance its adversarial robustness.

## Team Information
- **Team Name:** CRAG-MLAdversary

## Team Members
- **Guo, Jiawei**
  - **Email:** jiawei.guo@wsu.edu
  - **WSU ID:** 011854914

- **Azza Fadhel**
  - **Email:** azza.fadhel@wsu.edu
  - **WSU ID:** 11819844
  - **Contact:** The contact person

- **Stirewalt, Tashi Russell**
  - **Email:** tashi.stirewalt@wsu.edu
  - **WSU ID:** 011809728

- **Ugwu, Chibuike Emmanuel**
  - **Email:** chibuike.ugwu@wsu.edu
  - **WSU ID:** 11806472

## Usage information
- Make sure Node.js is locally installed. Download available at https://nodejs.org/en/download
- Execute node -v in terminal to make sure it downloaded and pathed properly
- Naviagte to frontend/src/api
- Use ```npm install``` -> ```npm run serve``` to launch front end
- Run .sql to create a database

## To make a prediction with the classifier
- !python3  prediction.py  image_path  model_path
- Below is an example:
- !python3  prediction.py  /home/cugwu_dg/cpts-528-project/528Project/dog_example.jpg  /home/cugwu_dg/cpts-528-project/528Project/model.pth
