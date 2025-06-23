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
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, GRU
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Neural network models will be skipped.")
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
    Data loader for PADS dataset - optimized for preprocessed data format
    """
    def __init__(self, data_path, use_preprocessed=True):
        self.data_path = Path(data_path)
        self.use_preprocessed = use_preprocessed
        
        # Set up paths based on data format
        if self.use_preprocessed:
            self.patients_dir = self.data_path.parent / 'patients'
            self.questionnaire_dir = self.data_path.parent / 'questionnaire'
            self.movement_dir = self.data_path / 'movement'
            self.file_list_path = self.data_path / 'file_list.csv'
        else:
            self.patients_dir = self.data_path / 'patients'
            self.questionnaire_dir = self.data_path / 'questionnaire'
            self.movement_dir = self.data_path / 'movement'
        
    def load_patient_data(self):
        """Load patient metadata"""
        patient_files = list(self.patients_dir.glob('patient_*.json'))
        patients = []
        
        for file in patient_files:
            with open(file, 'r') as f:
                patient_data = json.load(f)
                patients.append(patient_data)
        
        return pd.DataFrame(patients)
    
    def get_preprocessed_channel_info(self):
        """Get channel information for preprocessed data based on documentation"""
        tasks = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", 
                "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]
        wrists = ["Left", "Right"]
        
        # Documentation shows inconsistency - try both naming conventions
        # The preprocessed examples show "Acceleration" and "Rotation"
        # But the sensor list shows "Accelerometer" and "Gyroscope"
        sensors_v1 = ["Acceleration", "Rotation"]  # From preprocessed examples
        sensors_v2 = ["Accelerometer", "Gyroscope"]  # From sensor list
        
        axes = ["X", "Y", "Z"]
        
        # Try the first naming convention (from examples)
        channels_v1 = []
        for task in tasks:
            for wrist in wrists:
                for sensor in sensors_v1:
                    for axis in axes:
                        channel_name = f"{task}_{wrist}_{sensor}_{axis}"
                        channels_v1.append(channel_name)
        
        # Also create the alternative naming convention
        channels_v2 = []
        for task in tasks:
            for wrist in wrists:
                for sensor in sensors_v2:
                    for axis in axes:
                        channel_name = f"{task}_{wrist}_{sensor}_{axis}"
                        channels_v2.append(channel_name)
        
        return channels_v1, channels_v2, tasks, wrists, sensors_v1, axes
    
    def load_preprocessed_data(self, subject_id):
        """Load preprocessed binary data for a subject with enhanced debugging"""
        if not self.use_preprocessed:
            raise ValueError("This method is only for preprocessed data")
        
        # Try different file naming patterns - the debug output shows they use _ml.bin suffix
        possible_files = [
            self.movement_dir / f'{subject_id:03d}_ml.bin',  # This is the correct pattern!
            self.movement_dir / f'{subject_id:03d}.bin',
            self.movement_dir / f'{subject_id}.bin',
            self.movement_dir / f'subject_{subject_id:03d}.bin'
        ]
        
        # Debug: Check what files actually exist for this subject
        if subject_id <= 3:  # Only debug first few patients
            print(f"Debug: Looking for files for subject {subject_id}")
            print(f"Movement directory: {self.movement_dir}")
            print(f"Directory exists: {self.movement_dir.exists()}")
            
            if self.movement_dir.exists():
                all_files = list(self.movement_dir.glob('*'))
                print(f"Total files in movement dir: {len(all_files)}")
                
                # Show first few files
                sample_files = [f.name for f in all_files[:10]]
                print(f"Sample files: {sample_files}")
                
                # Look for files containing this subject ID
                subject_files = [f for f in all_files if str(subject_id).zfill(3) in f.name or str(subject_id) in f.name]
                print(f"Files potentially for subject {subject_id}: {[f.name for f in subject_files]}")
        
        bin_file = None
        for file_path in possible_files:
            if file_path.exists():
                bin_file = file_path
                if subject_id <= 3:
                    print(f"Found file: {bin_file}")
                break
        
        if bin_file is None:
            if subject_id <= 3:
                print(f"No binary file found for subject {subject_id}")
            return None
        
        try:
            # Load binary data as float32 (as specified in documentation)
            data = np.fromfile(bin_file, dtype=np.float32)
            
            if subject_id <= 3:
                print(f"Loaded {len(data)} float32 values from {bin_file.name}")
            
            if len(data) == 0:
                if subject_id <= 3:
                    print(f"Empty file for subject {subject_id}")
                return None
            
            # According to documentation: 132 channels total (11 tasks * 2 wrists * 6 sensors)
            # Each task has 1024 samples (10.24 seconds at 100Hz)
            expected_channels = 132
            
            if subject_id <= 3:
                print(f"Trying to reshape {len(data)} values with {expected_channels} channels")
                print(f"Data length divisible by 132? {len(data) % 132 == 0}")
            
            if len(data) % expected_channels == 0:
                total_samples = len(data) // expected_channels
                reshaped_data = data.reshape(total_samples, expected_channels)
                if subject_id <= 3:
                    print(f"Successfully reshaped to {reshaped_data.shape}")
                return reshaped_data
            
            # Fallback: try other common configurations
            for channel_count in [66, 264, 12, 6]:
                if len(data) % channel_count == 0:
                    total_samples = len(data) // channel_count
                    reshaped_data = data.reshape(total_samples, channel_count)
                    if subject_id <= 3:
                        print(f"Using fallback: {channel_count} channels, shape {reshaped_data.shape}")
                    return reshaped_data
            
            if subject_id <= 3:
                print(f"Could not determine proper shape for subject {subject_id} data length {len(data)}")
                # Try to understand the data structure
                print(f"Data type: {data.dtype}")
                print(f"First 10 values: {data[:10]}")
                print(f"Data range: {data.min()} to {data.max()}")
            return None
            
        except Exception as e:
            if subject_id <= 3:
                print(f"Error loading preprocessed data for subject {subject_id}: {e}")
            return None
    
    def extract_task_data(self, preprocessed_data, task_name, wrist=None):
        """Extract specific task and wrist data from preprocessed format"""
        if preprocessed_data is None:
            return None
        
        channels_v1, channels_v2, tasks, wrists, sensors, axes = self.get_preprocessed_channel_info()
        
        # Try both naming conventions
        for channels in [channels_v1, channels_v2]:
            # Find channels for the specific task (and optionally wrist)
            task_indices = []
            for i, channel in enumerate(channels):
                if channel.startswith(task_name):
                    if wrist is None or f"_{wrist}_" in channel:
                        task_indices.append(i)
            
            if task_indices and len(task_indices) <= preprocessed_data.shape[1]:
                # Extract data for this task
                task_data = preprocessed_data[:, task_indices]
                return task_data
        
        print(f"Warning: Could not find channels for task {task_name}")
        return None
    
    def load_questionnaire_data(self):
        """Load questionnaire responses"""
        questionnaire_files = list(self.questionnaire_dir.glob('questionnaire_response_*.json'))
        questionnaire_data = []
        
        for file in questionnaire_files:
            with open(file, 'r') as f:
                data = json.load(f)
                questionnaire_data.append(data)
        
        return questionnaire_data

class FeatureExtractor:
    """Enhanced feature extraction for time series sensor data"""
    
    @staticmethod
    def time_domain_features(signal):
        """Extract time domain features"""
        if len(signal) == 0:
            return {}
        
        features = {}
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['var'] = np.var(signal)
        
        # Handle edge cases for skewness and kurtosis
        if features['std'] > 0:
            features['skewness'] = stats.skew(signal)
            features['kurtosis'] = stats.kurtosis(signal)
        else:
            features['skewness'] = 0
            features['kurtosis'] = 0
        
        # Zero crossings
        features['zero_crossings'] = len(np.where(np.diff(np.sign(signal)))[0])
        
        # Peak detection
        if len(signal) > 1:
            peaks, _ = find_peaks(signal)
            features['peak_count'] = len(peaks)
        else:
            features['peak_count'] = 0
        
        # Additional features
        features['median'] = np.median(signal)
        features['q25'] = np.percentile(signal, 25)
        features['q75'] = np.percentile(signal, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        return features
    
    @staticmethod
    def frequency_domain_features(signal, sampling_rate=100):
        """Extract frequency domain features"""
        if len(signal) < 2:
            return {}
        
        features = {}
        
        # FFT
        fft_vals = fft(signal)
        fft_freqs = fftfreq(len(signal), 1/sampling_rate)
        fft_magnitude = np.abs(fft_vals)
        
        # Keep only positive frequencies
        positive_freq_idx = fft_freqs > 0
        if not np.any(positive_freq_idx):
            return {}
        
        fft_freqs = fft_freqs[positive_freq_idx]
        fft_magnitude = fft_magnitude[positive_freq_idx]
        
        if len(fft_magnitude) == 0 or np.sum(fft_magnitude) == 0:
            return {}
        
        features['dominant_freq'] = fft_freqs[np.argmax(fft_magnitude)]
        features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude)
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        
        # Spectral entropy (with numerical stability)
        normalized_magnitude = fft_magnitude / np.sum(fft_magnitude)
        features['spectral_entropy'] = -np.sum(normalized_magnitude * np.log2(normalized_magnitude + 1e-12))
        
        # Additional frequency features
        features['spectral_std'] = np.sqrt(np.sum(((fft_freqs - features['spectral_centroid'])**2) * normalized_magnitude))
        features['spectral_skewness'] = np.sum(((fft_freqs - features['spectral_centroid'])**3) * normalized_magnitude) / (features['spectral_std']**3 + 1e-12)
        
        return features
    
    @staticmethod
    def wavelet_features(signal, wavelet='db4', levels=3):
        """Extract wavelet features with error handling"""
        if len(signal) < 4:  # Minimum length for wavelet decomposition
            return {}
        
        features = {}
        try:
            coeffs = pywt.wavedec(signal, wavelet, level=levels)
            
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
                    features[f'wavelet_std_level_{i}'] = np.std(coeff)
                    features[f'wavelet_mean_level_{i}'] = np.mean(coeff)
                else:
                    features[f'wavelet_energy_level_{i}'] = 0
                    features[f'wavelet_std_level_{i}'] = 0
                    features[f'wavelet_mean_level_{i}'] = 0
                    
        except Exception as e:
            print(f"Warning: Wavelet decomposition failed: {e}")
            
        return features
    
    def extract_all_features(self, signal, sampling_rate=100):
        """Extract all features from a signal with error handling"""
        if signal is None or len(signal) == 0:
            return {}
        
        # Ensure signal is 1D
        if signal.ndim > 1:
            signal = signal.flatten()
        
        # Remove NaN and infinite values
        signal = signal[np.isfinite(signal)]
        
        if len(signal) == 0:
            return {}
        
        features = {}
        
        # Time domain features
        try:
            features.update(self.time_domain_features(signal))
        except Exception as e:
            print(f"Warning: Time domain feature extraction failed: {e}")
        
        # Frequency domain features
        try:
            features.update(self.frequency_domain_features(signal, sampling_rate))
        except Exception as e:
            print(f"Warning: Frequency domain feature extraction failed: {e}")
        
        # Wavelet features
        try:
            features.update(self.wavelet_features(signal))
        except Exception as e:
            print(f"Warning: Wavelet feature extraction failed: {e}")
        
        return features

class PADSBaselineModels:
    """Baseline models for PADS dataset classification"""
    
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
    
    def evaluate_traditional_models(self, X, y, cv_folds=5):
        """Evaluate traditional ML models with cross-validation"""
        if len(X) == 0 or len(y) == 0:
            print("No data available for evaluation")
            return
        
        self.initialize_models()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            try:
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
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                print("-" * 50)
    
    def print_results_summary(self):
        """Print summary of all results"""
        if not self.results:
            print("No results to display. Run evaluation first.")
            return
        
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

def extract_patient_metadata_features(patient_row):
    """Extract features from patient metadata with better error handling"""
    metadata_features = []
    
    # Numerical features with default values
    numerical_cols = ['age', 'height', 'weight', 'age_at_diagnosis']
    for col in numerical_cols:
        if col in patient_row and pd.notna(patient_row[col]):
            try:
                metadata_features.append(float(patient_row[col]))
            except (ValueError, TypeError):
                metadata_features.append(0.0)
        else:
            metadata_features.append(0.0)
    
    # Categorical features (one-hot encoded)
    # Gender
    gender = patient_row.get('gender', '').lower()
    metadata_features.append(1.0 if gender == 'male' else 0.0)
    metadata_features.append(1.0 if gender == 'female' else 0.0)
    
    # Handedness
    handedness = patient_row.get('handedness', '').lower()
    metadata_features.append(1.0 if handedness == 'right' else 0.0)
    metadata_features.append(1.0 if handedness == 'left' else 0.0)
    
    # Kinship appearance (boolean features)
    metadata_features.append(1.0 if patient_row.get('appearance_in_kinship') is True else 0.0)
    metadata_features.append(1.0 if patient_row.get('appearance_in_first_grade_kinship') is True else 0.0)
    
    # Effect of alcohol on tremor
    alcohol_effect = patient_row.get('effect_of_alcohol_on_tremor', 'Unknown')
    metadata_features.append(1.0 if alcohol_effect == 'Improvement' else 0.0)
    metadata_features.append(1.0 if alcohol_effect == 'No effect' else 0.0)
    metadata_features.append(1.0 if alcohol_effect == 'Worsening' else 0.0)
    metadata_features.append(1.0 if alcohol_effect == 'Unknown' else 0.0)
    
    return metadata_features

def main():
    """Main pipeline for PADS baseline evaluation"""
    
    # Configuration
    use_preprocessed = True
    data_path = "/Users/kiranshay/projects/Parkinsons-ET/pads-dataset/preprocessed"  # Update this path
    
    print(f"Using {'PREPROCESSED' if use_preprocessed else 'RAW'} data format")
    print(f"Data path: {data_path}")
    
    # Initialize components
    loader = PADSDataLoader(data_path, use_preprocessed=use_preprocessed)
    feature_extractor = FeatureExtractor()
    baseline_models = PADSBaselineModels()
    
    # Verify paths exist
    if not Path(data_path).exists():
        print(f"ERROR: Data path '{data_path}' does not exist!")
        return
    
    print("Loading PADS dataset...")
    
    # Load patient data
    patients_df = loader.load_patient_data()
    
    if len(patients_df) == 0:
        print("ERROR: No patient data loaded!")
        return
    
    print(f"Loaded data for {len(patients_df)} patients")
    
    # Check for condition column
    condition_col = 'condition'
    if condition_col not in patients_df.columns:
        print(f"ERROR: '{condition_col}' column not found in patient data")
        print(f"Available columns: {list(patients_df.columns)}")
        return
    
    print(f"Class distribution:")
    print(patients_df[condition_col].value_counts())
    
    # Feature extraction
    print(f"\nExtracting features from patients...")
    
    features_list = []
    labels = []
    successful_loads = 0
    failed_loads = 0
    
    # Select key tasks for feature extraction (to reduce dimensionality)
    key_tasks = ["Relaxed1", "StretchHold", "HoldWeight", "TouchNose"]
    
    for idx, patient in patients_df.iterrows():
        patient_id = int(patient['id'])
        condition = patient[condition_col]
        
        # Extract metadata features
        patient_features = extract_patient_metadata_features(patient)
        initial_feature_count = len(patient_features)
        
        # Load preprocessed movement data
        preprocessed_data = loader.load_preprocessed_data(patient_id)
        
        if preprocessed_data is not None:
            # Extract features from key tasks
            features_extracted = False
            for task in key_tasks:
                task_data = loader.extract_task_data(preprocessed_data, task)
                
                if task_data is not None and task_data.shape[1] > 0:
                    # Extract features from each channel of this task
                    for channel_idx in range(min(12, task_data.shape[1])):  # Limit to 12 channels per task
                        signal = task_data[:, channel_idx]
                        
                        # Skip first 50 samples (0.5 seconds) as recommended in documentation
                        if len(signal) > 50:
                            signal = signal[50:]
                        
                        if len(signal) > 10:  # Ensure we have enough data
                            features = feature_extractor.extract_all_features(signal)
                            if features:  # Only add if features were extracted
                                patient_features.extend(list(features.values()))
                                features_extracted = True
            
            # Only include if we have sufficient features
            if len(patient_features) > initial_feature_count:  # More than just metadata
                features_list.append(patient_features)
                labels.append(condition)
                successful_loads += 1
            else:
                failed_loads += 1
        else:
            failed_loads += 1
        
        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(patients_df)} patients...")
    
    print(f"\nData loading summary:")
    print(f"Successfully processed: {successful_loads} patients")
    print(f"Failed to process: {failed_loads} patients")
    
    if successful_loads == 0:
        print("ERROR: No patients processed successfully!")
        return
    
    # Convert to arrays
    X_features = np.array(features_list)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X_features.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Handle missing values
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Evaluate models
    print("\nEvaluating baseline models...")
    baseline_models.evaluate_traditional_models(X_features, y)
    
    # Print results
    baseline_models.print_results_summary()

if __name__ == "__main__":
    main()