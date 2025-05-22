import os
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import logging
from config import Config  

# Setup logging
logging.basicConfig(level=logging.INFO, filename='training.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

logger.info("Starting script")

# Load training data
try:
    train = pd.read_csv(Config.TRAIN_DATA_PATH)
    logger.info("Data loaded successfully")
except FileNotFoundError:
    logger.error(f"Error: '{Config.TRAIN_DATA_PATH}' not found")
    raise
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# Feature identification
original_features = [col for col in train.columns if col not in ['id', 'target']]
categorical_features = [col for col in train.columns if '_cat' in col]
numerical_features = [col for col in train.columns if col not in categorical_features + ['id', 'target']]

# Handle missing values
for col in numerical_features:
    median_val = train[col][train[col] != -1].median()
    train[col] = train[col].replace(-1, median_val)
logger.info("Missing values handled")

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(train[categorical_features])
cat_encoded_cols = [f"{col}_{val}" for i, col in enumerate(categorical_features) for val in encoder.categories_[i]]
cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_encoded_cols, index=train.index)
logger.info("Categorical features encoded")

# Prepare X and y
X = pd.concat([train[numerical_features], cat_encoded_df], axis=1)
y = train['target']
logger.info("X and y prepared")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
logger.info("Data split")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
logger.info("Features scaled")

# Train RandomForest
rf_model = RandomForestClassifier(random_state=Config.RANDOM_STATE, n_jobs=-1)
logger.info("Training RandomForest")
rf_model.fit(X_train, y_train)
# Save RandomForest model immediately
version = Config.VERSION
with open(f'models/rf_model_v{version}.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
logger.info("Saved RandomForest model")
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
logger.info(f"RandomForest CV scores: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
rf_accuracy = rf_model.score(X_val, y_val)

# Train MLP
ann_model = MLPClassifier(hidden_layer_sizes=Config.ANN_LAYERS, max_iter=Config.MAX_ITER, random_state=Config.RANDOM_STATE)
logger.info("Training MLP")
ann_model.fit(X_train_scaled, y_train)
# Save MLP model immediately
with open(f'models/ann_model_v{version}.pkl', 'wb') as f:
    pickle.dump(ann_model, f)
logger.info("Saved MLP model")
ann_cv_scores = cross_val_score(ann_model, X_train_scaled, y_train, cv=5)
logger.info(f"MLP CV scores: {ann_cv_scores.mean():.3f} (+/- {ann_cv_scores.std() * 2:.3f})")
ann_accuracy = ann_model.score(X_val_scaled, y_val)

# Save preprocessors and other artifacts
for file_name, obj in [
    ('models/encoder.pkl', encoder),
    ('models/scaler.pkl', scaler)
]:
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved {file_name}")

# Save accuracies and feature info
accuracies = {'rf_accuracy': rf_accuracy, 'ann_accuracy': ann_accuracy}
with open('models/model_accuracies.pkl', 'wb') as f:
    pickle.dump(accuracies, f)

medians = {col: train[col][train[col] != -1].median() for col in numerical_features}
feature_info = {
    'original_features': original_features,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'medians': medians
}
with open('models/feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)

logger.info("Training complete")