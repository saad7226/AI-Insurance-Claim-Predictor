class Config:
    SECRET_KEY = 'saad7223' 
    TRAIN_DATA_PATH = 'train.csv'
    MODEL_DIR = 'models'
    VERSION = '1.0.0'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    ANN_LAYERS = (100,)
    MAX_ITER = 500
    DEBUG = True  
    PORT = 5000