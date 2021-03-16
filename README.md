### Machine Learning Ensemble To Determine Tumor Benign or Malignant Status 

## Introduction
Welcome to our open source ensemble! This program is optimized to predict malignant or benign status of a tumor given quatitative measurements of the tumor cells nucleii.
 The intended users for this function are pathologists who have fine needle aspirations of breast cancer tumors. 

## The Motivation
The motivation behind this software is to attempt to increase relaibility of pathologist's Malignant/Benign designation, as the current accuracy is only about 80%(Sangha B, et al. CAR Journal. 2016. 67(3):284-9).
We thought that given the quantitative measurements that pathologists gather about the tumor's nucleii, we could predict (with better accuracy) the tumors status with an enemble model.
The reason we chose an ensemble approach is to have a "voting approach" approach to the issue, the most common output from each of our weak learners will be included as the ensemble's designation. 

## Build Status
[![Build Status](https://travis-ci.com/Breast-Cancer-Team/Final-Project.svg?branch=main)](https://travis-ci.com/github/Breast-Cancer-Team/Final-Project)

## Code Style
Code is PEP8 compliant. 

## Tech/Framework Used
Built with:
- Jupyter Notebook
- Sublime Text

## Features:
- command line accessible calling capabilities
- All ML algortihms used, saved as .py files
- a wrapping function that acquires the data from the user and outputs a prediction to the command line.
- a data splitting function that takes the users input data and cleans+pasrses it. 
 
## Use Cases: 
User: Pathologist examining fine needle aspirations of breast cancer tissue under a microscope for malignancy status.

Information Necessary:
- 'radius_mean',
- 'texture_mean',
- 'perimeter_mean',
- 'area_mean',
- 'smoothness_mean',
- 'compactness_mean',
- 'concavity_mean',
- 'concave points_mean',
- 'symmetry_mean',
- 'fractal_dimension_mean'

Responses: 
- Boolean of malignant or benign nature of tumor based on input information.
- Plot of data and predicted accuracy values.
  
Messages: 
- Dimensions out of bounds.
- Not enough information provided.
- Incorrect File Type

## Installation Instructions
In terminal, change directories to where the user would like the file to be saved. 
To download the functions simply put: git clone https://github.com/Breast-Cancer-Team/Final-Project.git

## Code Test Examples
For Averaging Ensemble
- Open terminal and move into the working directory of ensemble_wrapping_function.py
- in the command line input: python3 ensemble_wrapping_function.py -averaging
- when promted, supply filename.csv. (I.E. test_file.csv)
- files should output from here!

For stacking classifier:
- similar to above; Open terminal and move into the working directory of stacking.py
- in the command line input: python3 stacking.py -stacking
- when promted, supply filename.csv. (I.E. test_file.csv)
- files should output from here!
 

## Contributing Guidline
If you would like to contribute to this project, areas in which more contribution would be welcome is in the generations of additional weak learners to add to the voting population.
Additionally, a graphic user interface for the pathologists to easily access the program would also be helpful. The current documentation is PEP8 compliant and worked out of the GitHub Repository listed below. 

## Credits
https://github.com/Breast-Cancer-Team/Final-Project.git

## License 
MIT License - "A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code." 

MIT © Breast-Cancer-Team
