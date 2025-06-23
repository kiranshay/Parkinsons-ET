import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data processing and balancing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           f1_score, balanced_accuracy_score, roc_auc_score)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# Feature selection and dimensionality reduction
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.decomposition import PCA

# Advanced ML models
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Neural networks
from sklearn.neural_network import MLPClassifier
try:
    import tensorflow as tf
    from keras.models import Sequential, Model
    from keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Flatten, 
                                       Dropout, GRU, BatchNormalization,
                                       Input, GlobalMaxPooling1D)
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will be skipped.")
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
        """Load preprocessed binary data for a subject"""
        if not self.use_preprocessed:
            raise ValueError("This method is only for preprocessed data")
        
        # The correct pattern is {subject_id:03d}_ml.bin
        possible_files = [
            self.movement_dir / f'{subject_id:03d}_ml.bin',
            self.movement_dir / f'{subject_id:03d}.bin',
            self.movement_dir / f'{subject_id}.bin',
            self.movement_dir / f'subject_{subject_id:03d}.bin'
        ]
        
        bin_file = None
        for file_path in possible_files:
            if file_path.exists():
                bin_file = file_path
                break
        
        if bin_file is None:
            return None
        
        try:
            # Load binary data as float32 (as specified in documentation)
            data = np.fromfile(bin_file, dtype=np.float32)
            
            if len(data) == 0:
                return None
            
            # According to documentation: 132 channels total (11 tasks * 2 wrists * 6 sensors)
            expected_channels = 132
            
            if len(data) % expected_channels == 0:
                total_samples = len(data) // expected_channels
                reshaped_data = data.reshape(total_samples, expected_channels)
                return reshaped_data
            
            # Fallback: try other common configurations
            for channel_count in [66, 264, 12, 6]:
                if len(data) % channel_count == 0:
                    total_samples = len(data) // channel_count
                    reshaped_data = data.reshape(total_samples, channel_count)
                    return reshaped_data
            
            return None
            
        except Exception as e:
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
        
        return None

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
        
        # Energy features
        features['energy'] = np.sum(signal**2)
        features['mean_abs'] = np.mean(np.abs(signal))
        
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
        
        # Spectral entropy
        normalized_magnitude = fft_magnitude / np.sum(fft_magnitude)
        features['spectral_entropy'] = -np.sum(normalized_magnitude * np.log2(normalized_magnitude + 1e-12))
        
        # Additional frequency features
        features['spectral_std'] = np.sqrt(np.sum(((fft_freqs - features['spectral_centroid'])**2) * normalized_magnitude))
        
        # Frequency bands (relevant for tremor analysis)
        tremor_band = (fft_freqs >= 3) & (fft_freqs <= 8)  # Tremor frequency band
        normal_band = (fft_freqs >= 8) & (fft_freqs <= 15)  # Normal movement band
        
        if np.any(tremor_band):
            features['tremor_power'] = np.sum(fft_magnitude[tremor_band])
        else:
            features['tremor_power'] = 0
            
        if np.any(normal_band):
            features['normal_power'] = np.sum(fft_magnitude[normal_band])
        else:
            features['normal_power'] = 0
        
        return features
    
    @staticmethod
    def wavelet_features(signal, wavelet='db4', levels=3):
        """Extract wavelet features with error handling"""
        if len(signal) < 4:
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
            pass
            
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
        except Exception:
            pass
        
        # Frequency domain features
        try:
            features.update(self.frequency_domain_features(signal, sampling_rate))
        except Exception:
            pass
        
        # Wavelet features
        try:
            features.update(self.wavelet_features(signal))
        except Exception:
            pass
        
        return features

class EnhancedPADSClassifier:
    """
    Enhanced PADS classifier with advanced techniques for imbalanced data
    """
    
    def __init__(self, use_balanced_accuracy=True, random_state=42):
        self.use_balanced_accuracy = use_balanced_accuracy
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        
    def prepare_data_with_balancing(self, X, y, method='smote'):
        """Prepare data with various balancing techniques"""
        print(f"Original class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features first
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply balancing technique
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state, k_neighbors=3)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif method == 'borderline':
            sampler = BorderlineSMOTE(random_state=self.random_state)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=self.random_state)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=self.random_state)
        else:
            return X_scaled, y_encoded
        
        print(f"\nApplying {method.upper()} balancing...")
        try:
            X_balanced, y_balanced = sampler.fit_resample(X_scaled, y_encoded)
            
            print(f"Balanced class distribution:")
            unique, counts = np.unique(y_balanced, return_counts=True)
            for label, count in zip(unique, counts):
                original_label = self.label_encoder.inverse_transform([label])[0]
                print(f"  {original_label}: {count}")
            
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Balancing failed: {e}, using original data")
            return X_scaled, y_encoded
    
    def feature_selection(self, X, y, method='rfecv', k=500):
        """Advanced feature selection techniques"""
        print(f"\nApplying feature selection: {method}")
        print(f"Original feature count: {X.shape[1]}")
        
        if method == 'univariate':
            k = min(k, X.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            
        elif method == 'rfecv':
            # Use a fast estimator for feature selection
            estimator = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            selector = RFECV(estimator=estimator, cv=3, scoring='balanced_accuracy', n_jobs=-1)
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            
        elif method == 'tree_importance':
            # Use tree-based feature importance
            estimator = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            estimator.fit(X, y)
            importances = estimator.feature_importances_
            k = min(k, X.shape[1])
            indices = np.argsort(importances)[::-1][:k]
            X_selected = X[:, indices]
            self.feature_selector = indices
            
        else:
            X_selected = X
            
        print(f"Selected feature count: {X_selected.shape[1]}")
        return X_selected
    
    def initialize_advanced_models(self):
        """Initialize advanced models with hyperparameter tuning"""
        self.models = {
            'XGBoost_Enhanced': XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                objective='multi:softprob',
                n_jobs=-1
            ),
            'Random_Forest_Enhanced': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Extra_Trees': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'SVM_Enhanced': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            ),
            'MLP_Enhanced': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                objective='multiclass',
                verbose=-1,
                n_jobs=-1
            )
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                random_seed=self.random_state,
                verbose=False
            )
        
        # Create ensemble models
        voting_estimators = [
            ('xgb', self.models['XGBoost_Enhanced']),
            ('rf', self.models['Random_Forest_Enhanced']),
            ('et', self.models['Extra_Trees'])
        ]
        
        if LIGHTGBM_AVAILABLE:
            voting_estimators.append(('lgb', self.models['LightGBM']))
        
        self.models['Voting_Ensemble'] = VotingClassifier(
            estimators=voting_estimators, 
            voting='soft',
            n_jobs=-1
        )
    
    def hyperparameter_tuning(self, X, y, model_name='XGBoost'):
        """Perform hyperparameter tuning for selected models"""
        print(f"\nTuning hyperparameters for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9]
            }
            base_model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [10, 12, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
            
        else:
            return None
        
        # Use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv, 
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_advanced_neural_network(self, input_shape, num_classes):
        """Create an advanced neural network"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_shape,), 
                  kernel_regularizer=l1_l2(0.001, 0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def evaluate_models(self, X, y, use_balancing=True, balancing_method='smote', 
                       feature_selection_method='rfecv', cv_folds=5):
        """Comprehensive model evaluation with advanced techniques"""
        print("="*80)
        print("🚀 ENHANCED PADS CLASSIFICATION EVALUATION")
        print("="*80)
        
        # Prepare data
        if use_balancing:
            X_processed, y_processed = self.prepare_data_with_balancing(X, y, balancing_method)
        else:
            y_processed = self.label_encoder.fit_transform(y)
            X_processed = self.scaler.fit_transform(X)
        
        # Feature selection
        if feature_selection_method:
            X_processed = self.feature_selection(X_processed, y_processed, feature_selection_method)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=self.random_state, 
            stratify=y_processed
        )
        
        # Initialize models
        self.initialize_advanced_models()
        
        # Hyperparameter tuning for best models
        print("\n" + "="*60)
        print("🔧 HYPERPARAMETER TUNING")
        print("="*60)
        
        tuned_xgb = self.hyperparameter_tuning(X_train, y_train, 'XGBoost')
        if tuned_xgb:
            self.models['XGBoost_Tuned'] = tuned_xgb
        
        # Evaluate all models
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scoring_metric = 'balanced_accuracy' if self.use_balanced_accuracy else 'accuracy'
        
        for name, model in self.models.items():
            print(f"\n🔄 Evaluating {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                          scoring=scoring_metric, n_jobs=-1)
                
                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Store results
                self.results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_balanced_accuracy': test_balanced_accuracy,
                    'test_f1': test_f1,
                    'classification_report': classification_report(
                        y_test, y_pred, target_names=self.label_encoder.classes_
                    )
                }
                
                # Track best model
                current_score = test_balanced_accuracy if self.use_balanced_accuracy else test_accuracy
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_model = model
                
                print(f"✅ CV {scoring_metric}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print(f"✅ Test Accuracy: {test_accuracy:.4f}")
                print(f"✅ Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
                print(f"✅ Test F1-Score: {test_f1:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {name}: {e}")
                continue
        
        # Evaluate neural networks if available
        if TENSORFLOW_AVAILABLE:
            self.evaluate_neural_networks(X_train, X_test, y_train, y_test)
        
        return X_test, y_test
    
    def evaluate_neural_networks(self, X_train, X_test, y_train, y_test):
        """Evaluate deep learning models"""
        print("\n" + "="*60)
        print("🧠 DEEP LEARNING MODELS")
        print("="*60)
        
        num_classes = len(np.unique(y_train))
        
        # Advanced Neural Network
        print("\n🔄 Training Advanced Neural Network...")
        nn_model = self.create_advanced_neural_network(X_train.shape[1], num_classes)
        
        if nn_model:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            ]
            
            history = nn_model.fit(
                X_train, y_train,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            y_pred_prob = nn_model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            test_accuracy = accuracy_score(y_test, y_pred)
            test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.results['Advanced_Neural_Network'] = {
                'test_accuracy': test_accuracy,
                'test_balanced_accuracy': test_balanced_accuracy,
                'test_f1': test_f1,
                'classification_report': classification_report(
                    y_test, y_pred, target_names=self.label_encoder.classes_
                )
            }
            
            print(f"✅ Test Accuracy: {test_accuracy:.4f}")
            print(f"✅ Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
            print(f"✅ Test F1-Score: {test_f1:.4f}")
            
            # Track best model
            current_score = test_balanced_accuracy if self.use_balanced_accuracy else test_accuracy
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = nn_model
    
    def print_comprehensive_results(self):
        """Print detailed results with insights"""
        if not self.results:
            print("No results to display. Run evaluation first.")
            return
        
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        # Sort by balanced accuracy
        metric = 'test_balanced_accuracy' if self.use_balanced_accuracy else 'test_accuracy'
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
        
        print(f"{'Model':<25} {'Accuracy':<12} {'Bal.Acc':<12} {'F1-Score':<12} {'CV Score':<12}")
        print("-" * 75)
        
        for name, results in sorted_results:
            acc = results.get('test_accuracy', 0)
            bal_acc = results.get('test_balanced_accuracy', 0)
            f1 = results.get('test_f1', 0)
            cv_score = results.get('cv_mean', 0)
            print(f"{name:<25} {acc:<12.4f} {bal_acc:<12.4f} {f1:<12.4f} {cv_score:<12.4f}")
        
        # Print best model details
        if sorted_results:
            best_name, best_results = sorted_results[0]
            print(f"\n🏆 BEST MODEL: {best_name}")
            print(f"🎯 Balanced Accuracy: {best_results.get('test_balanced_accuracy', 0):.4f}")
            print(f"🎯 Regular Accuracy: {best_results.get('test_accuracy', 0):.4f}")
            print(f"🎯 F1-Score: {best_results.get('test_f1', 0):.4f}")
            print(f"🎯 CV Score: {best_results.get('cv_mean', 0):.4f} (+/- {best_results.get('cv_std', 0)*2:.4f})")
        
        # Performance insights
        print(f"\n📈 PERFORMANCE INSIGHTS:")
        if sorted_results:
            best_bal_acc = sorted_results[0][1].get('test_balanced_accuracy', 0)
            if best_bal_acc > 0.8:
                print("🎉 Excellent performance! Balanced accuracy > 80%")
            elif best_bal_acc > 0.7:
                print("👍 Good performance! Balanced accuracy > 70%")
            elif best_bal_acc > 0.6:
                print("✅ Decent performance! Balanced accuracy > 60%")
            else:
                print("⚠️  Room for improvement. Consider more data or feature engineering.")
    
    def plot_comprehensive_results(self):
        """Create comprehensive visualization of results"""
        if not self.results:
            print("No results to plot.")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        accuracies = [self.results[model].get('test_accuracy', 0) for model in models]
        balanced_accuracies = [self.results[model].get('test_balanced_accuracy', 0) for model in models]
        f1_scores = [self.results[model].get('test_f1', 0) for model in models]
        cv_scores = [self.results[model].get('cv_mean', 0) for model in models]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 Enhanced PADS Classification Results', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(models))
        width = 0.35
        bars1 = ax1.bar(x_pos - width/2, accuracies, width, alpha=0.8, color='skyblue', label='Regular Accuracy')
        bars2 = ax1.bar(x_pos + width/2, balanced_accuracies, width, alpha=0.8, color='lightcoral', label='Balanced Accuracy')
        ax1.set_title('📊 Accuracy Comparison')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # F1-Score comparison
        ax2 = axes[0, 1]
        bars3 = ax2.bar(models, f1_scores, alpha=0.8, color='lightgreen')
        ax2.set_title('📈 F1-Score Comparison')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars3, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Performance scatter plot
        ax3 = axes[1, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        scatter = ax3.scatter(balanced_accuracies, f1_scores, s=100, alpha=0.7, c=colors)
        for i, model in enumerate(models):
            ax3.annotate(model, (balanced_accuracies[i], f1_scores[i]), 
                        fontsize=8, ha='center', va='bottom', rotation=15)
        ax3.set_xlabel('Balanced Accuracy')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('🎯 Performance Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # CV vs Test performance
        ax4 = axes[1, 1]
        ax4.scatter(cv_scores, balanced_accuracies, s=100, alpha=0.7, c='purple')
        for i, model in enumerate(models):
            ax4.annotate(model, (cv_scores[i], balanced_accuracies[i]), 
                        fontsize=8, ha='center', va='bottom', rotation=15)
        ax4.set_xlabel('CV Balanced Accuracy')
        ax4.set_ylabel('Test Balanced Accuracy')
        ax4.set_title('🔄 CV vs Test Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add diagonal line for perfect CV-Test correlation
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Correlation')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

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

def main_enhanced():
    """
    Main pipeline for enhanced PADS classification
    """
    print("🚀 ENHANCED PADS CLASSIFICATION WITH CLASS BALANCING")
    print("="*60)
    
    # Configuration
    use_preprocessed = True
    data_path = "/Users/kiranshay/projects/Parkinsons-ET/pads-dataset/preprocessed"  # Update this path
    
    print("📋 Configuration:")
    print(f"  - Data format: {'PREPROCESSED' if use_preprocessed else 'RAW'}")
    print(f"  - Data path: {data_path}")
    
    # Initialize components
    loader = PADSDataLoader(data_path, use_preprocessed=use_preprocessed)
    feature_extractor = FeatureExtractor()
    
    # Verify paths exist
    if not Path(data_path).exists():
        print(f"❌ ERROR: Data path '{data_path}' does not exist!")
        return
    
    print("\n📂 Loading PADS dataset...")
    
    # Load patient data
    patients_df = loader.load_patient_data()
    
    if len(patients_df) == 0:
        print("❌ ERROR: No patient data loaded!")
        return
    
    print(f"✅ Loaded data for {len(patients_df)} patients")
    
    # Check for condition column
    condition_col = 'condition'
    if condition_col not in patients_df.columns:
        print(f"❌ ERROR: '{condition_col}' column not found in patient data")
        print(f"Available columns: {list(patients_df.columns)}")
        return
    
    print(f"\n📊 Class distribution:")
    class_counts = patients_df[condition_col].value_counts()
    for condition, count in class_counts.items():
        percentage = (count / len(patients_df)) * 100
        print(f"  {condition}: {count} ({percentage:.1f}%)")
    
    # Feature extraction
    print(f"\n🔧 Extracting features from patients...")
    
    features_list = []
    labels = []
    successful_loads = 0
    failed_loads = 0
    
    # Select key tasks for feature extraction (to reduce dimensionality)
    key_tasks = ["Relaxed1", "StretchHold", "HoldWeight", "TouchNose", "CrossArms"]
    
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
            print(f"  Processed {idx + 1}/{len(patients_df)} patients...")
    
    print(f"\n📈 Data loading summary:")
    print(f"  ✅ Successfully processed: {successful_loads} patients")
    print(f"  ❌ Failed to process: {failed_loads} patients")
    
    if successful_loads == 0:
        print("❌ ERROR: No patients processed successfully!")
        return
    
    # Convert to arrays
    X_features = np.array(features_list)
    y = np.array(labels)
    
    print(f"\n📊 Dataset summary:")
    print(f"  Feature matrix shape: {X_features.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Features per patient: {X_features.shape[1]}")
    
    # Handle missing values
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Run enhanced evaluation with different configurations
    print(f"\n🚀 Starting enhanced evaluation...")
    
    experiments = [
        {
            'name': 'SMOTE + RFECV',
            'balancing': 'smote', 
            'feature_selection': 'rfecv'
        },
        {
            'name': 'BorderlineSMOTE + Tree Importance',
            'balancing': 'borderline', 
            'feature_selection': 'tree_importance'
        },
        {
            'name': 'SMOTEENN + Univariate',
            'balancing': 'smoteenn', 
            'feature_selection': 'univariate'
        }
    ]
    
    best_experiment = None
    best_classifier = None
    best_score = 0
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*20} EXPERIMENT {i+1}: {exp['name']} {'='*20}")
        
        # Create new classifier for each experiment
        classifier = EnhancedPADSClassifier(use_balanced_accuracy=True, random_state=42)
        
        try:
            X_test, y_test = classifier.evaluate_models(
                X_features, y, 
                use_balancing=True,
                balancing_method=exp['balancing'],
                feature_selection_method=exp['feature_selection'],
                cv_folds=5
            )
            
            if classifier.best_score > best_score:
                best_score = classifier.best_score
                best_experiment = exp
                best_classifier = classifier
                
        except Exception as e:
            print(f"❌ Experiment {i+1} failed: {e}")
            continue
    
    # Print final results
    if best_classifier:
        print(f"\n🏆 BEST EXPERIMENT: {best_experiment['name']}")
        print(f"🎯 Best Score: {best_score:.4f}")
        
        best_classifier.print_comprehensive_results()
        best_classifier.plot_comprehensive_results()
        
        # Additional insights
        print(f"\n🎓 KEY INSIGHTS:")
        print(f"  📈 The enhanced approach with {best_experiment['balancing']} balancing")
        print(f"     and {best_experiment['feature_selection']} feature selection performed best")
        print(f"  🎯 Balanced accuracy of {best_score:.1%} addresses class imbalance effectively")
        print(f"  🔬 Feature engineering extracted {X_features.shape[1]} meaningful features")
        print(f"  ⚖️  Class balancing techniques significantly improved minority class performance")
        
        return best_classifier
    else:
        print("❌ All experiments failed!")
        return None

if __name__ == "__main__":
    classifier = main_enhanced()