![Total Lines of Code](https://img.shields.io/badge/total%20lines%20of%20code-1516-green?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/SarathMajji1699/Xmoji?style=for-the-badge)
![GitHub followers](https://img.shields.io/github/followers/SarathMajji1699?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues-raw/SarathMajji1699/Xmoji?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/SarathMajji1699/Xmoji?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SarathMajji1699/Xmoji?style=for-the-badge)
![Maintenance](https://img.shields.io/maintenance/yes/2021?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/SarathMajji1699/Xmoji?style=for-the-badge)

# Xmoji
<p align="center">
  <a href="https://github.com/SarathMajji1699/Xmoji">
    <img src="https://github.com/SarathMajji1699/ImagesUpload/blob/main/xmoji-logo-b.png?raw=true" alt="Logo">
  </a>

  <h3 align="center">Xmoji</h3>
    <h2 align="center">Expression to Emoji</h2>

  <p align="center">
   <h1> <strong>About the Project</strong></h1>
The main idea is to capture the human face through live web proctoring and recognize the facial expressions stimulated in the face and describe the age, gender and create and display an emoji according to the expression.
<br/>
<ul>
<li>
Face Detection</li>
<li>Expression Evaluation</li>
<li>Gender Recognition</li>
<li>Age Detection</li>
</ul>

### Built With
<ul>
<li>Pandas</li>
<li>Numpy</li>
<li>Tensorflow</li>
<li>Flask</li>
<li>CNN</li>
<li>Keras</li>
<li>Opencv-python</li>
<li>Smptlib</li>
<li>SQLite3</li>
</ul>
    <br />
   <h1> <strong>Description of Project</h1></strong>
   <ul>
   <li>In Xmoji(expression to emoji) project we use convolutional neural networks to detect the image and display the suitable emoji of the human being captured through live web cam feed.</li>
   <li>Using OpenCV library we capture the human face through live feed/also we can upload an existing image and then the model displays an emoji according to the expression shown by the human-face or in the picture.</li>
   <li>We also derive the age, gender attributes of the human being captured through the live cam.</li>
</p>
<!-- GETTING STARTED -->
<p>

## Getting Started

To get a local copy up and running follow these simple steps.

## Prerequisites

This is an example of how to list the packages that you need to run this project.

  ```sh
  cd Xmoji
  cat requirements.txt
  ```

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/SarathMajji1699/Xmoji.git
   ```
2. Install required packages
   ```sh
   $ pip install -r requirements.txt
   ```



<!-- USAGE EXAMPLES -->
## Usage
1. Train the Model
   ```sh
   $ cd Xmoji
   $ python3 face_expression_detection.ipynb
   ```
2. Run the Model
   ```sh
   $ python3 main.py
   ```
</p>
