# Interactive Machine Learning Dashboard for iCAAV Vehicle Data
## Project Links

- GitHub Repository: [Capstone ICAAV Project App](https://github.com/muluwerkderebe-StatisticalAnalyst/Capstone-ICAAV-Project-App)
- Live Application: [Streamlit Deployment](https://capstone-icaav-project-app-qyoge5zfckgdab5sjs49ep.streamlit.app)
- Carleton University: [Carleton University](https://carleton.ca)
- Algonquin College: [Algonquin College](https://www.algonquincollege.com)
- iCAAV Core: [iCAAV Research](https://carleton.ca/biomechatronics/)

An interactive Streamlit-based machine learning platform developed as a Capstone Project within the Business Intelligence Systems Infrastructure program at Algonquin College in collaboration with the Intelligent Connected Assistive & Autonomous Vehicles (iCAAV) Core at Carleton University.

This application provides an end-to-end workflow for:

- Data loading and preprocessing
- Feature engineering
- Supervised machine learning
- Semi-supervised and unsupervised learning
- Deep learning (TensorFlow/Keras)
- Real-time visualization and annotation
- Driver behavior analysis using vehicle sensor data

---

## Project Overview

This Capstone Project was developed to support multimodal vehicle data analysis, machine learning model development, data visualization, and real-time annotation for intelligent transportation and autonomous vehicle research.

The platform enables researchers and students to:

- Import and process large datasets
- Perform feature engineering and extraction
- Train and evaluate machine learning models
- Compare model performance
- Conduct clustering and semi-supervised learning
- Visualize data and annotations in real time
- Analyze driver behavior using vehicle sensor data

---

## Institutional Collaboration

This Capstone project was completed through the Business Intelligence Systems Infrastructure program at Algonquin College in collaboration with:

### Carleton University

**Intelligent Connected Assistive & Autonomous Vehicles (iCAAV) Core**

**Advanced Biomechatronics and Locomotion Laboratory**

Ottawa, Ontario, Canada

---

## Application Modules

### 1. Data Processing

#### Features

- CSV Import
- Excel Import
- MATLAB (.mat) Import
- Missing Value Handling
- Data Normalization
- Feature Scaling
- Feature Engineering
- Statistical Feature Extraction
- Windowing Operations
- Dataset Visualization

---

### 2. Supervised Machine Learning

#### Classical Machine Learning

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

#### Deep Learning

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)

#### Evaluation Metrics

##### Regression

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

##### Classification

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves

---

### 3. Semi-Supervised and Unsupervised Learning

#### Semi-Supervised Learning

- Self-Training
- Label Propagation
- Label Spreading

#### Unsupervised Learning

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

#### Additional Tools

- Cluster Visualization
- Data Quality Analysis
- Feature Exploration
- Missing Data Assessment

---

### 4. Real-Time Testing, Visualization, and Annotation

#### Features

- Real-Time Data Monitoring
- Live Visualization
- UDP Data Streaming
- Annotation Tools
- Label Management
- Event Marking
- Data Export
- Real-Time Model Testing

---

## Technology Stack

### Frontend

- Streamlit

### Data Processing

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-Learn

### Deep Learning

- TensorFlow
- Keras

### Visualization

- Matplotlib
- Seaborn

### Data Formats

- CSV
- XLSX
- XLS
- MATLAB (.mat)

---

## Project Structure

```text
Capstone-ICAAV-Project-App/
│
├── app.py
├── theme.py
├── branding.py
├── data_utils.py
├── udp_receiver.py
├── udp_sender.py
├── requirements.txt
│
├── assets/
│   ├── icaav_logo.png
│   └── carleton_logo.png
│
├── pages/
│   ├── 1_Data_Processing.py
│   ├── 2_Supervised_Learning.py
│   ├── 3_SSL_Unsupervised.py
│   └── 4_RealTime_Annotation.py
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/muluwerkderebe-StatisticalAnalyst/Capstone-ICAAV-Project-App.git

cd Capstone-ICAAV-Project-App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using Streamlit Community Cloud.

To deploy:

1. Connect the GitHub repository to Streamlit Community Cloud.
2. Select the repository and branch.
3. Set the main file path to:

```text
app.py
```

4. Deploy the application.

---

## Research Applications

This platform can be used for:

- Driver Behavior Analysis
- Vehicle Dynamics Research
- Autonomous Vehicle Research
- Human-Machine Interaction Studies
- Sensor Fusion Projects
- Transportation Safety Research
- Biomechatronics Research
- Machine Learning Education and Research
- Intelligent Transportation Systems Research

---

## Authors

**Muluwerk Derebe**
**Zainularab Zarabi**
**Thituyetngan Nguyen**
**Mohamad Alsabbagh**

Business Intelligence Systems Infrastructure Student  
Algonquin College

Capstone Project Developer

Project completed in collaboration with:

- Intelligent Connected Assistive & Autonomous Vehicles (iCAAV) Core
- Advanced Biomechatronics and Locomotion Laboratory
- Carleton University

Ottawa, Ontario, Canada

---

## Acknowledgements

This project was developed as part of a Capstone Project within the Business Intelligence Systems Infrastructure program at Algonquin College in collaboration with the Intelligent Connected Assistive & Autonomous Vehicles (iCAAV) Core and the Advanced Biomechatronics and Locomotion Laboratory at Carleton University.

The author would like to acknowledge the guidance, mentorship, and support provided by faculty members, researchers, and project supervisors throughout the development of this platform.

---

## License

This project is intended for academic, educational, and research purposes.

© 2026 Muluwerk Derebe, Algonquin College, and collaborating research partners at Carleton University.
