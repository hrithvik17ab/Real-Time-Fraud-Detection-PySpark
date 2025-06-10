# Real-Time Fraud Detection System with PySpark & Machine Learning

## Project Overview
This project demonstrates a prototype for a real-time financial fraud detection system. It simulates an end-to-end pipeline, starting from synthetic data generation, training an anomaly detection model, processing a live stream of transactions using PySpark, applying the model for real-time predictions, and visualizing the results through a simulated live dashboard and static reports.

## Key Features
- **Synthetic Data Generation:** Custom script to generate a configurable dataset of financial transactions (~30,000 samples) with embedded fraudulent patterns and engineered time-based features (transaction hour, day of the week).
- **Machine Learning Model:** Utilizes Scikit-learn's `IsolationForest` for anomaly detection. Includes data preprocessing steps like `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.
- **Real-Time Stream Processing:** Employs PySpark Streaming to simulate the ingestion and processing of live transaction data at a defined rate (1 event/sec).
- **Live Fraud Prediction:** Integrates the trained Scikit-learn model into the Spark stream via User-Defined Functions (UDFs) for real-time scoring and flagging of each transaction.
- **Behavioral Analytics:** Implements stateful windowed aggregations (e.g., transaction count per user over 5-minute sliding windows) using PySpark Streaming to provide contextual behavioral insights.
- **Monitoring & Visualization:**
    - Real-time console alerts for ML-detected fraud and significant user activity patterns.
    - A simulated live dashboard built with Plotly `FigureWidget` and Spark SQL (querying an in-memory table), dynamically updating with overall transaction volumes and fraud statistics.
    - A static summary plot generated using Plotly from aggregated data collected during the streaming simulation, allowing for post-run analysis.
- **Model Evaluation:** Comprehensive performance review of the batch-trained model, detailing accuracy, precision, recall, F1-score, a visual confusion matrix, and distribution of anomaly scores.

## Technologies Used
- **Programming Language:** Python
- **Big Data Framework:** Apache Spark (PySpark API)
    - PySpark SQL (for querying in-memory tables)
    - PySpark Streaming (for real-time data processing)
- **Machine Learning:** Scikit-learn
    - Model: `IsolationForest`
    - Preprocessing: `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`
    - Evaluation: `train_test_split`, `classification_report`, `confusion_matrix`, `accuracy_score`
- **Data Handling & Manipulation:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Visualization:** Plotly (for dynamic `FigureWidget` and static charts), Matplotlib, Seaborn (for confusion matrix & score distribution plots)
- **Environment:** Google Colab (developed and tested)

## Setup and How to Run
1.  **Environment:** This project is designed to be run in a Google Colab notebook.
2.  **Dependencies:** The notebook includes a cell at the beginning to install all necessary Python packages using `pip`. Key dependencies are listed in `requirements.txt`.
3.  **Execution:** Open the `.ipynb` notebook in Google Colab and run the cells sequentially from top to bottom.
    - The script is divided into logical sections, from dependency installation to final model evaluation.
    - The streaming simulation (Section 6) will run for approximately 2.5-3 minutes, displaying live updates.
    - A static summary plot will be generated in Section 7.
    - Model performance metrics will be displayed in Section 8.

## Project Structure (Sections in the Notebook)
1.  **Dependency Installation & Environment Setup:** Installs required libraries and initializes the SparkSession.
2.  **Synthetic Data Generation:** Creates the `batch_df_pd` DataFrame with transaction details and engineered time features.
3.  **Batch Model Training & Evaluation:** Preprocesses the batch data, trains the `IsolationForest` model, evaluates it on a test set, and saves the trained preprocessor and model artifacts.
4.  **Streaming Pipeline Definition:** Loads the saved model/preprocessor, defines UDFs for data generation and prediction, and sets up the PySpark streams.
5.  **Real-Time Alerting & Output:** Starts console-based streaming queries for fraud alerts and user behavior aggregates.
6.  **Monitoring & Visualization (Live Simulation):** Initiates a streaming query to populate an in-memory table with dashboard statistics and runs a Python loop to update a Plotly `FigureWidget` dynamically.
7.  **Static Plot of Collected Dashboard Data:** After the live simulation, queries the in-memory table and generates a persistent static Plotly chart.
8.  **Model Performance Review:** Provides a detailed breakdown of the batch-trained model's performance metrics on the test set.

## Results & Key Metrics
*(Based on a run with N_TOTAL_BATCH_SAMPLES = 30,000, Contamination = 0.15, and Time Features)*
- **Overall Model Accuracy (Test Set):** ~86.2%
- **Fraud Detection Performance (Test Set - Class 'Fraudulent (1)'):**
    - **Precision:** ~0.43
    - **Recall:** ~0.57
    - **F1-Score:** ~0.49
- The system successfully demonstrated real-time processing of transactions, application of the ML model, and dynamic updates to monitoring metrics. The inclusion of time-based features and tuning of the `contamination` parameter contributed to achieving a fraud recall of 0.57.

## Potential Improvements & Future Work
- **Advanced Feature Engineering:** Incorporate more complex features like user historical spending patterns, transaction velocity over various time windows, or graph-based features if relationships between entities were modeled.
- **Alternative Models:** Experiment with supervised ML models (e.g., Random Forest, XGBoost) if higher quality labels were available, or other anomaly detection techniques (e.g., Autoencoders, LOF).
- **Optimized State Management:** For more complex user profiling in streaming, explore PySpark's `mapGroupsWithState` or `flatMapGroupsWithState`.
- **Scalable Deployment:** For a production scenario, deploy on a proper Spark cluster and use robust message queues (like Kafka) for data ingestion and a dedicated dashboarding solution (e.g., Plotly Dash, Streamlit deployed as a web app, or BI tools).
- **Model Retraining & Concept Drift:** Implement strategies for periodic model retraining and adaptation to changing fraud patterns.

## Author
- **Hrithvik**
