import os
from azure.storage.blob import BlobServiceClient
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('app.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Azure Blob Storage configuration
connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'insurance-files'

# Function to download files from Blob Storage on demand
def download_from_blob(blob_name, local_path):
    if not os.path.exists(local_path):
        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(local_path, "wb") as f:
                blob_data = blob_client.download_blob()
                f.write(blob_data.readall())
            logger.info(f"Downloaded {blob_name} to {local_path} from Blob Storage.")
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            raise

# Create models directory if it doesn’t exist
os.makedirs('models', exist_ok=True)

# Load small, essential .pkl files at startup
try:
    download_from_blob('feature_info.pkl', 'models/feature_info.pkl')
    with open('models/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
        original_features = feature_info['original_features']
        categorical_features = feature_info['categorical_features']
        numerical_features = feature_info['numerical_features']
        medians = feature_info['medians']
    
    download_from_blob('model_accuracies.pkl', 'models/model_accuracies.pkl')
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
except Exception as e:
    logger.error(f"Error loading startup files: {e}")
    raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_analysis')
def data_analysis():
    try:
        # Download train.csv on demand
        download_from_blob('train.csv', 'train.csv')
        
        # Get number of rows and columns without loading the entire file
        with open('train.csv', 'r') as f:
            num_columns = len(f.readline().split(','))
            num_rows = sum(1 for line in f)
        
        # Load only the first 5 rows for the head
        train_head = pd.read_csv('train.csv', nrows=5)
        
        # Compute missing values in chunks
        chunk_size = 10000
        missing = pd.Series(0, index=train_head.columns)
        for chunk in pd.read_csv('train.csv', chunksize=chunk_size):
            missing += chunk.isnull().sum()
        
        # Compute class distribution in chunks
        class_dist = pd.Series(0, index=[0,1])
        for chunk in pd.read_csv('train.csv', usecols=['target'], chunksize=chunk_size):
            class_dist += chunk['target'].value_counts()
        
        summary = {
            'head': train_head.to_html(classes='table table-striped', header="true"),
            'missing': missing.to_dict(),
            'rows': num_rows,
            'columns': num_columns,
            'class_dist': class_dist.to_dict()
        }
        
        # Plot class distribution
        plt.figure(figsize=(8, 6))
        plt.bar(class_dist.index, class_dist.values)
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
        # Download train.csv on demand
        download_from_blob('train.csv', 'train.csv')
        
        # Load a sample of 100 rows for the heatmap
        sample_data = pd.read_csv('train.csv', nrows=100)
        missing_count = sample_data.isnull().sum().sum()
        if missing_count == 0:
            results = "<p>No missing values in sample.</p>"
        else:
            results = "<p>Missing values present in sample.</p>"
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(sample_data.isnull(), cmap='viridis', cbar=True)
        plt.title('Missing Values Heatmap (Sample)')
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
            return render_template('visualization.html', error="No numerical features available.")
        
        # Download train.csv on demand
        download_from_blob('train.csv', 'train.csv')
        
        # Load a sample of 1000 rows for correlation matrix
        sample_data = pd.read_csv('train.csv', usecols=numerical_features, nrows=1000)
        numeric_data = sample_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return render_template('visualization.html', error="No numeric data available.")
        
        # Select top 20 features with highest variance
        top_features = numeric_data.var().nlargest(20).index
        numeric_data = numeric_data[top_features]
        
        plt.figure(figsize=(12, 10))
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr), k=1)
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0,
                    annot_kws={"size": 8}, square=True, cbar_kws={"shrink": .5})
        plt.title('Correlation Matrix of Top 20 Numerical Features (Sample)')
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
            input_data = {}
            for feat in original_features:
                value = request.form.get(feat)
                if not value:
                    if feat in numerical_features:
                        input_data[feat] = medians.get(feat, 0)
                    else:
                        input_data[feat] = -1
                else:
                    if feat in numerical_features:
                        value_float = float(value)
                        min_val, max_val = numerical_ranges.get(feat, (None, None))
                        if min_val is not None and max_val is not None:
                            if value_float < min_val or value_float > max_val:
                                raise ValueError(f"Value for {feat} out of range [{min_val}, {max_val}]")
                        input_data[feat] = value_float
                    else:
                        value_int = int(value)
                        if value_int not in valid_categories.get(feat, []):
                            raise ValueError(f"Invalid category for {feat}: {valid_categories.get(feat, [])}")
                        input_data[feat] = value_int

            input_df = pd.DataFrame([input_data])
            for col in numerical_features:
                if pd.isna(input_df[col].iloc[0]):
                    input_df[col] = medians.get(col, 0)

            # Download and load encoder on demand
            download_from_blob('encoder.pkl', 'models/encoder.pkl')
            with open('models/encoder.pkl', 'rb') as f:
                loaded_encoder = pickle.load(f)
            
            if categorical_features:
                cat_encoded = loaded_encoder.transform(input_df[categorical_features])
                cat_encoded_cols = [f"{col}_{val}" for i, col in enumerate(categorical_features) for val in loaded_encoder.categories_[i]]
                cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_encoded_cols)
                input_df = pd.concat([input_df.drop(categorical_features, axis=1), cat_encoded_df], axis=1)

            model_features = numerical_features + cat_encoded_cols
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[model_features]
            
            # Download and load RF model on demand
            rf_model_path = f'models/rf_model_v{Config.VERSION}.pkl'
            download_from_blob(f'rf_model_v{Config.VERSION}.pkl', rf_model_path)
            with open(rf_model_path, 'rb') as f:
                loaded_rf_model = pickle.load(f)
            rf_pred = loaded_rf_model.predict(input_df)[0]
            rf_result = "High Risk" if rf_pred == 1 else "Low Risk"

            # Download and load scaler on demand
            download_from_blob('scaler.pkl', 'models/scaler.pkl')
            with open('models/scaler.pkl', 'rb') as f:
                loaded_scaler = pickle.load(f)
            input_scaled = loaded_scaler.transform(input_df)

            # Download and load ANN model on demand
            ann_model_path = f'models/ann_model_v{Config.VERSION}.pkl'
            download_from_blob(f'ann_model_v{Config.VERSION}.pkl', ann_model_path)
            with open(ann_model_path, 'rb') as f:
                loaded_ann_model = pickle.load(f)
            ann_pred = loaded_ann_model.predict(input_scaled)[0]
            ann_result = "High Risk" if ann_pred >= 0.5 else "Low Risk"

            prediction = {
                'rf': rf_result,
                'ann': ann_result,
                'rf_acc': rf_accuracy,
                'ann_acc': ann_accuracy
            }
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
        {'name': 'Muhammad Arsalan Aslam', 'desc': 'ML Engineer', 'image': 'member3.jpg'}
    ]
    return render_template('about.html', team=team)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', Config.PORT))
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=port)