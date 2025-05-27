import os
import gdown  # Replace requests with gdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from config import Config
import logging

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)

# Setup logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('app.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Function to download files from Google Drive using gdown
def download_from_drive(file_id, local_path):
    if not os.path.exists(local_path):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False)
            # Validate the downloaded file is not an HTML page
            with open(local_path, 'rb') as f:
                first_bytes = f.read(10)
                if first_bytes.startswith(b'<!DOCTYPE') or first_bytes.startswith(b'<html'):
                    os.remove(local_path)
                    raise ValueError(f"Downloaded file {local_path} is an HTML page, not the expected pickle file.")
            logger.info(f"Downloaded {local_path} from Google Drive.")
        except Exception as e:
            logger.error(f"Failed to download {local_path}: {e}")
            raise

# File IDs from Google Drive links
file_ids = {
    f'rf_model_v{Config.VERSION}.pkl': '1n1BKl3YZmVeYvzA4gzv4Ay7SapbJ44qv',
    f'ann_model_v{Config.VERSION}.pkl': '1W_DvTaLT5BM3icqgC9wnlYx-vjASGvbg',
    'encoder.pkl': '1Ec0fuA51vWqp5rLYnGi2GB4sTvAQxFXy',
    'scaler.pkl': '1WwhZFmSF9X4qtjdc5o9jRj5J80DRW8W1',
    'feature_info.pkl': '1HoiKpOdE-jhykfvO_sEkC2wqvURVA6w-',
    'model_accuracies.pkl': '14hJ2yi3wLLfXr0J-ORlzAeFsY7L178lD'
}

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download all .pkl files if they don't exist locally
for filename, file_id in file_ids.items():
    local_path = os.path.join('models', filename)
    download_from_drive(file_id, local_path)

# Load models and preprocessors
try:
    with open(f'models/rf_model_v{Config.VERSION}.pkl', 'rb') as f:
        loaded_rf_model = pickle.load(f)
        logger.info("RF model loaded successfully")
    with open(f'models/ann_model_v{Config.VERSION}.pkl', 'rb') as f:
        loaded_ann_model = pickle.load(f)
        logger.info("ANN model loaded successfully")
    with open('models/encoder.pkl', 'rb') as f:
        loaded_encoder = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    with open('models/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
        original_features = feature_info['original_features']
        categorical_features = feature_info['categorical_features']
        numerical_features = feature_info['numerical_features']
        medians = feature_info['medians']
    with open('models/model_accuracies.pkl', 'rb') as f:
        accuracies = pickle.load(f)
        rf_accuracy = accuracies['rf_accuracy']
        ann_accuracy = accuracies['ann_accuracy']
except FileNotFoundError as e:
    logger.error(f"Missing file: {e}")
    raise
except pickle.UnpicklingError as e:
    logger.error(f"Unpickling error: {e}")
    raise

# Define model features
cat_encoded_cols = [f"{col}_{val}" for i, col in enumerate(categorical_features) for val in loaded_encoder.categories_[i]]
model_features = numerical_features + cat_encoded_cols

# Load data for analysis
try:
    train_data = pd.read_csv('train.csv')
    logger.info("train.csv loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Failed to load train.csv: {e}")
    train_data = pd.DataFrame(columns=original_features)

numerical_ranges = {col: (train_data[col].min(), train_data[col].max()) for col in numerical_features if col in train_data.columns}
valid_categories = {col: list(loaded_encoder.categories_[i]) for i, col in enumerate(categorical_features)}

# Rest of your routes remain unchanged
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_analysis')
def data_analysis():
    try:
        summary = {
            'head': train_data.head().to_html(classes='table table-striped', header="true"),
            'missing': train_data.isnull().sum().to_dict(),
            'rows': train_data.shape[0],
            'columns': train_data.shape[1],
            'class_dist': train_data['target'].value_counts().to_dict()
        }
        plt.figure(figsize=(8, 6))
        train_data['target'] = train_data['target'].astype(int)
        sns.countplot(data=train_data, x='target')
        plt.title('Class Distribution')
        plt.xlabel('Target (0: No Claim, 1: Claim)')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No Claim', 'Claim'])
        plt.savefig('static/images/class_dist.png', bbox_inches='tight')
        plt.close()
        return render_template('data_analysis.html', summary=summary)
    except Exception as e:
        logger.error(f"Error in data_analysis: {e}")
        return render_template('data_analysis.html', error=str(e))

@app.route('/preprocessing')
def preprocessing():
    try:
        missing_count = train_data.isnull().sum().sum()
        if missing_count == 0:
            results = "<p>No missing values detected in the dataset. Preprocessing steps (imputation and encoding) were applied, but no imputation was necessary.</p>"
        else:
            results = "<p>Imputation and encoding applied. The heatmap below shows missing values (yellow) before preprocessing.</p>"

        plt.figure(figsize=(10, 6))
        sns.heatmap(train_data.isnull(), cmap='viridis', cbar=True)
        plt.title('Missing Values Heatmap (Before)')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.savefig('static/images/missing_heatmap.png', bbox_inches='tight')
        plt.close()
        return render_template('preprocessing.html', results=results)
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return render_template('preprocessing.html', error=str(e))

@app.route('/visualization')
def visualization():
    try:
        if not numerical_features:
            return render_template('visualization.html', error="No numerical features available for correlation matrix.")

        numeric_data = train_data[numerical_features].select_dtypes(include=[np.number])
        if numeric_data.empty:
            return render_template('visualization.html', error="No numeric data available for correlation matrix.")

        numeric_data = numeric_data.loc[:, numeric_data.var().nlargest(20).index]

        plt.figure(figsize=(12, 10))
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr), k=1)
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0,
                    annot_kws={"size": 8}, square=True, cbar_kws={"shrink": .5})
        plt.title('Correlation Matrix of Top 20 Numerical Features', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig('static/images/corr_matrix.png', bbox_inches='tight', dpi=300)
        plt.close()

        return render_template('visualization.html')
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        return render_template('visualization.html', error=str(e))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    error = None
    prediction = None
    try:
        if request.method == 'POST':
            logger.info("Received POST request for prediction")
            logger.info(f"Form data: {request.form}")
            input_data = {}
            for feat in original_features:
                value = request.form.get(feat)
                logger.info(f"Processing feature: {feat}, value: {value}")
                if not value:
                    if feat in numerical_features:
                        input_data[feat] = medians.get(feat, 0)
                    else:
                        input_data[feat] = -1
                else:
                    if feat in numerical_features:
                        try:
                            value_float = float(value)
                            min_val, max_val = numerical_ranges.get(feat, (None, None))
                            if min_val is not None and max_val is not None:
                                if value_float < min_val or value_float > max_val:
                                    raise ValueError(f"Value for {feat} ({value_float}) is out of range [{min_val}, {max_val}]")
                            input_data[feat] = value_float
                        except ValueError as e:
                            raise ValueError(str(e))
                    else:
                        try:
                            value_int = int(value)
                        except ValueError:
                            raise ValueError(f"Invalid input for {feat}: {value}. Must be an integer.")
                        if value_int not in valid_categories.get(feat, []):
                            raise ValueError(f"Invalid category for {feat}. Valid values are: {valid_categories.get(feat, [])}")
                        input_data[feat] = value_int

            logger.info("Input data prepared: %s", input_data)
            input_df = pd.DataFrame([input_data])
            logger.info("input_df shape: %s", input_df.shape)

            for col in numerical_features:
                if pd.isna(input_df[col].iloc[0]):
                    input_df[col] = medians.get(col, 0)

            if categorical_features:
                try:
                    cat_encoded = loaded_encoder.transform(input_df[categorical_features])
                    cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_encoded_cols)
                    input_df = pd.concat([input_df.drop(categorical_features, axis=1), cat_encoded_df], axis=1)
                except ValueError as e:
                    raise ValueError(f"Encoding failed: {e}")

            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[model_features]
            logger.info("Final input_df shape: %s", input_df.shape)

            rf_pred = loaded_rf_model.predict(input_df)[0]
            rf_result = "High Risk" if rf_pred == 1 else "Low Risk"

            input_scaled = loaded_scaler.transform(input_df)
            ann_pred = loaded_ann_model.predict(input_scaled)[0]
            ann_result = "High Risk" if ann_pred >= 0.5 else "Low Risk"

            prediction = {
                'rf': rf_result,
                'ann': ann_result,
                'rf_acc': rf_accuracy,
                'ann_acc': ann_accuracy
            }
            logger.info("Prediction successful: RF=%s, ANN=%s", rf_result, ann_result)

        return render_template('prediction.html', prediction=prediction, original_features=original_features, rf_accuracy=rf_accuracy, ann_accuracy=ann_accuracy, error=error)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        error = str(e)
        return render_template('prediction.html', prediction=None, original_features=original_features, rf_accuracy=rf_accuracy, ann_accuracy=ann_accuracy, error=error)

@app.route('/about')
def about():
    team = [
        {'name': 'Muhammad Saad Zafar', 'desc': 'Lead Developer', 'image': 'member1.jpg'},
        {'name': 'Muhammad Humayun Farasat', 'desc': 'Data Analyst', 'image': 'member2.jpg'},
        {'name': 'Muhammad Arsalan Aslam ', 'desc': 'ML Engineer', 'image': 'member3.jpg'}
    ]
    return render_template('about.html', team=team)

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=Config.PORT)