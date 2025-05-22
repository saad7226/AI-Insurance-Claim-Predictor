AI Insurance Predictor - Spring 2025

Welcome to the AI Insurance Predictor, a semester project developed for the Spring 2025 semester under the guidance of Lecturer Mustajab Hussain. This project leverages machine learning to predict insurance claim risks for automotive insurance customers, providing a fair and efficient solution for both insurance firms and policyholders. 

Project Overview
The AI Insurance Predictor is a Flask-based web application designed to assess the likelihood of an insurance customer filing a claim. Utilizing the Porto Seguro dataset, the application employs various machine learning algorithms to deliver accurate predictions. The project encompasses data preprocessing, model training, visualization, and deployment on a public cloud service, aiming to enhance transparency and accessibility in insurance risk assessment.
Features

Data Analysis: Summarizes dataset statistics, missing values, class distribution, and descriptive analytics.
Preprocessing: Handles missing values, encodes categorical features, manages class imbalance, and offers a downloadable cleaned dataset.
Visualization: Displays target variable distribution, missing value heatmaps, feature distributions, correlation matrices, and group-wise comparisons.
Prediction: Provides a form for inputting customer data, displays risk classification (High/Low), and includes model evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve).
Explainability: Features SHAP or LIME visualizations for feature contribution and local predictions.
About Us: Showcases team member details, including names, optional images, descriptions, and links to GitHub/LinkedIn.

Installation
To set up the project locally, follow these steps:

Clone the Repository
git clone https://github.com/your-username/ai-insurance-predictor.git
cd ai-insurance-predictor


Install DependenciesEnsure you have Python 3.8+ installed. Install the required libraries:
pip install -r requirements.txt


Prepare the Dataset

Download the Porto Seguro dataset (train.csv and test.csv) from the provided source or reference link.
Place the files in the data/ directory.


Run the Application

Set up the Flask environment:export FLASK_APP=app.py
flask run


Alternatively, run directly:python app.py


Open your browser and navigate to http://127.0.0.1:5000/.



Usage

Access the web application via the navigation bar:
Home: Welcome page with a "Get Started" button.
Data Analysis: View dataset summaries and statistics.
Preprocessing: Explore preprocessing steps and results.
Visualization: Interact with data visualizations.
Prediction: Input customer data for risk predictions.
About Us: Learn about the team.


The application is hosted publicly on a free cloud service ( Render). Check the repository for the live demo link.

Project Structure
ai-insurance-predictor/
│
├── /data/                # Contains train.csv and test.csv
├── /static/              # CSS, JS, and images
├── /templates/           # HTML templates
├── app.py                # Main Flask application file
├── requirements.txt      # List of dependencies
├── README.md             # This file


Technologies Used

Backend: Python, Flask
Frontend: HTML, CSS, Bootstrap, Jinja2
Machine Learning/Preprocessing: pandas, numpy, scikit-learn
Visualization: matplotlib, seaborn, plotly
Deployment: Render,

Models
The project uses the following machine learning algorithms:

Random Forest
Artificial Neural Network (ANN)

Dataset

Source: Porto Seguro
Files: train.csv (with target: 1 for claim, 0 for no claim), test.csv (without target)
Features: Binary (bin), categorical (cat), continuous/ordinal (scaled), with prefixes (ind, reg, car, calc)
Missing Values: Represented as -1

Deliverables

Web Application: Deployed Flask app with multiple pages.
GitHub Repository: Contains code, README, dataset reference, and performance metrics.


Contribution
This project was developed by Muhammad Saad Zafar,Muhammad Hamayun Farasat,Muhammad Arsalan Aslam. Contributions include code development, data analysis, visualization, and documentation.

Contact
For questions or collaboration, reach out via email saadzafar0505650@gmail.com

Acknowledgments
We thank Lecturer Mustajab Hussain for assigning this insightful project and Porto Seguro for providing the dataset.


Outputs
![image](https://github.com/user-attachments/assets/225f043f-d30a-409a-a5bd-9647084d72d0)
![image](https://github.com/user-attachments/assets/0ef15996-0483-481a-8d7f-470dd2cd7603)



