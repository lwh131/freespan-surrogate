from sklearn.preprocessing import StandardScaler
import joblib

class DataScaler:
    """A simple wrapper for StandardScaler to save/load."""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)
        return self
        
    def transform(self, data):
        return self.scaler.transform(data)
        
    def fit_transform(self, data):
        return self.scaler.fit_transform(data)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def save(self, filepath):
        joblib.dump(self.scaler, filepath)

    @staticmethod
    def load(filepath):
        scaler_instance = DataScaler()
        scaler_instance.scaler = joblib.load(filepath)
        return scaler_instance
