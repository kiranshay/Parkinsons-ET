import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data processing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Traditional ML models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Neural networks
from sklearn.neural_network import MLPClassifier
try:
    import tensorflow as tf
    # Try different import patterns for better compatibility
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, GRU
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        try:
            # Alternative import pattern
            from tensorflow import keras
            Sequential = keras.models.Sequential
            Dense = keras.layers.Dense
            LSTM = keras.layers.LSTM
            Conv1D = keras.layers.Conv1D
            MaxPooling1D = keras.layers.MaxPooling1D
            Flatten = keras.layers.Flatten
            Dropout = keras.layers.Dropout
            GRU = keras.layers.GRU
            Adam = keras.optimizers.Adam
            EarlyStopping = keras.callbacks.EarlyStopping
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            print("TensorFlow/Keras not available. Neural network models will be skipped.")
            TENSORFLOW_AVAILABLE = False
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    print("Neural network models will be skipped.")
    TENSORFLOW_AVAILABLE = False

# Feature extraction
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import pywt

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class PADSDataLoader:
    """
    Data loader for PADS dataset
    """
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.patients = []
        self.movement_data = []
        self.questionnaire_data = []
        
    def load_patient_data(self):
        """Load patient metadata"""
        patient_files = list((self.data_path / 'patients').glob('patient_*.json'))
        
        for file in patient_files:
            with open(file, 'r') as f:
                patient_data = json.load(f)
                self.patients.append(patient_data)
        
        return pd.DataFrame(self.patients)
    
    def load_movement_data(self):
        """Load movement observation data"""
        movement_files = list((self.data_path / 'movement').glob('observation_*.json'))
        
        for file in movement_files:
            with open(file, 'r') as f:
                movement_data = json.load(f)
                self.movement_data.append(movement_data)
        
        return self.movement_data
    
    def load_timeseries_data(self, subject_id, task_name, wrist):
        """Load specific time series data"""
        filename = f"{subject_id:03d}_{task_name}_{wrist}Wrist.txt"
        file_path = self.data_path / 'movement' / 'timeseries' / filename
        
        if file_path.exists():
            return pd.read_csv(file_path, header=None).values
        else:
            return None
    
    def load_questionnaire_data(self):
        """Load questionnaire responses"""
        questionnaire_files = list((self.data_path / 'questionnaire').glob('questionnaire_response_*.json'))
        
        for file in questionnaire_files:
            with open(file, 'r') as f:
                questionnaire_data = json.load(f)
                self.questionnaire_data.append(questionnaire_data)
        
        return self.questionnaire_data

class FeatureExtractor:
    """
    Feature extraction for time series sensor data
    """
    
    @staticmethod
    def time_domain_features(signal):
        """Extract time domain features"""
        features = {}
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['zero_crossings'] = len(np.where(np.diff(np.sign(signal)))[0])
        
        # Peak detection
        peaks, _ = find_peaks(signal)
        features['peak_count'] = len(peaks)
        
        return features
    
    @staticmethod
    def frequency_domain_features(signal, sampling_rate=100):
        """Extract frequency domain features"""
        features = {}
        
        # FFT
        fft_vals = fft(signal)
        fft_freqs = fftfreq(len(signal), 1/sampling_rate)
        fft_magnitude = np.abs(fft_vals)
        
        # Keep only positive frequencies
        positive_freq_idx = fft_freqs > 0
        fft_freqs = fft_freqs[positive_freq_idx]
        fft_magnitude = fft_magnitude[positive_freq_idx]
        
        features['dominant_freq'] = fft_freqs[np.argmax(fft_magnitude)]
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        features['spectral_entropy'] = -np.sum((fft_magnitude/np.sum(fft_magnitude)) * 
                                              np.log2(fft_magnitude/np.sum(fft_magnitude) + 1e-12))
        
        return features
    
    @staticmethod
    def wavelet_features(signal, wavelet='db4', levels=4):
        """Extract wavelet features"""
        features = {}
        coeffs = pywt.wavedec(signal, wavelet, level=levels)
        
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
            features[f'wavelet_std_level_{i}'] = np.std(coeff)
        
        return features
    
    def extract_all_features(self, signal, sampling_rate=100):
        """Extract all features from a signal"""
        features = {}
        features.update(self.time_domain_features(signal))
        features.update(self.frequency_domain_features(signal, sampling_rate))
        features.update(self.wavelet_features(signal))
        
        return features

class PADSBaselineModels:
    """
    Baseline models for PADS dataset classification
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def initialize_models(self):
        """Initialize all baseline models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Voting classifier
        voting_estimators = [
            ('rf', self.models['Random Forest']),
            ('xgb', self.models['XGBoost']),
            ('svm', self.models['SVM'])
        ]
        self.models['Voting Classifier'] = VotingClassifier(estimators=voting_estimators, voting='hard')
    
    def create_lstm_model(self, input_shape, num_classes):
        """Create LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_cnn_model(self, input_shape, num_classes):
        """Create 1D CNN model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def create_gru_model(self, input_shape, num_classes):
        """Create GRU model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def evaluate_traditional_models(self, X, y, cv_folds=5):
        """Evaluate traditional ML models with cross-validation"""
        self.initialize_models()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
            
            # Fit model for detailed evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store results
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred, average='weighted'),
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=self.label_encoder.classes_)
            }
            
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Test F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
            print("-" * 50)
    
    def evaluate_neural_networks(self, X_sequences, y, cv_folds=5):
        """Evaluate neural network models"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping neural network evaluation.")
            return
            
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        # Prepare data for neural networks
        X_sequences = np.array(X_sequences)
        input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        
        # Models to evaluate
        nn_models = {
            'LSTM': self.create_lstm_model(input_shape, num_classes),
            'CNN-1D': self.create_cnn_model(input_shape, num_classes),
            'GRU': self.create_gru_model(input_shape, num_classes)
        }
        
        for name, model in nn_models.items():
            if model is None:
                continue
                
            print(f"Evaluating {name}...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            
            self.results[name] = {
                'test_accuracy': test_accuracy,
                'test_f1': f1_score(y_test, y_pred, average='weighted'),
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=self.label_encoder.classes_)
            }
            
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
            print("-" * 50)
    
    def print_results_summary(self):
        """Print summary of all results"""
        print("\n" + "="*60)
        print("BASELINE MODELS PERFORMANCE SUMMARY")
        print("="*60)
        
        # Sort by test accuracy
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].get('test_accuracy', 0), 
                              reverse=True)
        
        print(f"{'Model':<20} {'Test Accuracy':<15} {'Test F1-Score':<15}")
        print("-" * 50)
        
        for name, results in sorted_results:
            acc = results.get('test_accuracy', 0)
            f1 = results.get('test_f1', 0)
            print(f"{name:<20} {acc:<15.4f} {f1:<15.4f}")
    
    def plot_results(self):
        """Plot comparison of model performances"""
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return
        
        # Extract data for plotting
        models = list(self.results.keys())
        accuracies = [self.results[model].get('test_accuracy', 0) for model in models]
        f1_scores = [self.results[model].get('test_f1', 0) for model in models]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Comparison - Test Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1-Score plot
        bars2 = ax2.bar(models, f1_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model Comparison - Test F1-Score')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Example usage and main pipeline
def main():
    """
    Main pipeline for PADS baseline evaluation
    """
    # Initialize components
    data_path = "/Users/kiranshay/projects/Parkinsons-ET/pads-dataset"  # ***Update this path as necessary*** 
                                                                        # Debugging print statements should help you if incorrect
    loader = PADSDataLoader(data_path)
    feature_extractor = FeatureExtractor()
    baseline_models = PADSBaselineModels()
    
    print("Loading PADS dataset...")
    
    # Debug: Check if the path exists
    if not Path(data_path).exists():
        print(f"ERROR: Dataset path '{data_path}' does not exist!")
        print(f"Current working directory: {Path.cwd()}")
        print("Please check the path to your dataset.")
        return
    
    # Debug: Check directory structure
    print(f"Dataset path exists: {data_path}")
    print(f"Contents of dataset directory:")
    for item in Path(data_path).iterdir():
        print(f"  - {item.name}")
    
    # Check if expected subdirectories exist
    expected_dirs = ['patients', 'movement', 'questionnaire']
    for dir_name in expected_dirs:
        dir_path = Path(data_path) / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob('*')))
            print(f"  {dir_name}/ directory exists with {file_count} files")
        else:
            print(f"  WARNING: {dir_name}/ directory not found!")
    
    # Load data
    patients_df = loader.load_patient_data()
    
    if len(patients_df) == 0:
        print("\nERROR: No patient data loaded!")
        print("This could be due to:")
        print("1. Incorrect dataset path")
        print("2. Missing 'patients' directory")
        print("3. No patient_*.json files in the patients directory")
        
        # Check patients directory specifically
        patients_dir = Path(data_path) / 'patients'
        if patients_dir.exists():
            patient_files = list(patients_dir.glob('patient_*.json'))
            print(f"\nFound {len(patient_files)} patient_*.json files in patients directory")
            if len(patient_files) > 0:
                print("Sample files:")
                for i, file in enumerate(patient_files[:3]):
                    print(f"  - {file.name}")
                    
                # Try to load and inspect one patient file
                try:
                    with open(patient_files[0], 'r') as f:
                        sample_patient = json.load(f)
                    print(f"\nSample patient data structure:")
                    for key in sample_patient.keys():
                        print(f"  - {key}: {sample_patient[key]}")
                except Exception as e:
                    print(f"Error reading sample patient file: {e}")
        return
    
    print(f"Loaded data for {len(patients_df)} patients")
    
    # Debug: Check what columns we actually have
    print(f"Patient dataframe columns: {list(patients_df.columns)}")
    
    # Check for the condition column or similar
    if 'condition' in patients_df.columns:
        print(f"Class distribution:")
        print(patients_df['condition'].value_counts())
    else:
        print("'condition' column not found. Looking for similar columns...")
        possible_condition_cols = [col for col in patients_df.columns if 'condition' in col.lower() or 'diagnosis' in col.lower() or 'class' in col.lower()]
        if possible_condition_cols:
            print(f"Found similar columns: {possible_condition_cols}")
            # Use the first similar column
            condition_col = possible_condition_cols[0]
            print(f"Using '{condition_col}' as condition column:")
            print(patients_df[condition_col].value_counts())
        else:
            print("No condition-like column found. Available columns:")
            for col in patients_df.columns:
                print(f"  - {col}: {patients_df[col].iloc[0] if len(patients_df) > 0 else 'N/A'}")
            return
    
    # Feature extraction example
    # This is a simplified example - you'll need to adapt based on your specific requirements
    features_list = []
    labels = []
    sequences_list = []
    
    # Task names from the dataset
    task_names = ['Relaxed', 'StretchHold', 'HoldWeight', 'DrinkGlas', 
                  'CrossArms', 'TouchNose', 'Entrainment']
    
    for idx, patient in patients_df.iterrows():
        patient_id = int(patient['id'])
        condition = patient['condition']
        
        patient_features = []
        patient_sequences = []
        
        for task in task_names:
            for wrist in ['Left', 'Right']:
                # Load time series data
                ts_data = loader.load_timeseries_data(patient_id, task, wrist)
                
                if ts_data is not None:
                    # Extract features for each sensor channel
                    for channel_idx in range(ts_data.shape[1]):
                        signal = ts_data[:, channel_idx]
                        features = feature_extractor.extract_all_features(signal)
                        patient_features.extend(list(features.values()))
                    
                    # Store sequence data for neural networks
                    patient_sequences.append(ts_data)
        
        if patient_features:
            features_list.append(patient_features)
            labels.append(condition)
            sequences_list.append(patient_sequences)
    
    # Convert to arrays
    X_features = np.array(features_list)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X_features.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Evaluate traditional ML models
    print("\nEvaluating traditional ML models...")
    baseline_models.evaluate_traditional_models(X_features, y)
    
    # For neural networks, we need to prepare sequence data properly
    # This is a simplified version - you may need more sophisticated preprocessing
    if sequences_list:
        print("\nEvaluating neural network models...")
        # Convert sequences to fixed-length format (you may need to adjust this)
        max_length = min(1024, max(len(seq[0]) if seq else 0 for seq in sequences_list))
        X_sequences = []
        
        for sequences in sequences_list:
            if sequences:
                # Concatenate all sequences for a patient (simplified approach)
                concat_seq = np.concatenate(sequences, axis=1) if len(sequences) > 1 else sequences[0]
                if len(concat_seq) >= max_length:
                    X_sequences.append(concat_seq[:max_length])
                else:
                    # Pad if necessary
                    padded = np.pad(concat_seq, ((0, max_length - len(concat_seq)), (0, 0)), 'constant')
                    X_sequences.append(padded)
            else:
                # Create dummy sequence if no data
                X_sequences.append(np.zeros((max_length, 6)))  # 6 channels (3 accel + 3 gyro)
        
        baseline_models.evaluate_neural_networks(X_sequences, y)
    
    # Print results summary
    baseline_models.print_results_summary()
    
    # Plot results
    baseline_models.plot_results()

if __name__ == "__main__":
    main()