#!/usr/bin/env python3
"""
Focused validation script for SMOTE + Voting Classifier model
Tests for overfitting and ensures results are robust and generalizable
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut, GroupKFold,
    cross_val_score, validation_curve, learning_curve
)
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing classes (assuming they're in the same directory)
from enhanced_models import PADSDataLoader, FeatureExtractor, extract_patient_metadata_features

class SMOTEVotingValidator:
    """
    Focused validation for SMOTE + Voting Classifier model
    Tests multiple validation strategies to detect overfitting
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.best_model = None
        self.validation_results = {}
        
    def create_smote_voting_classifier(self):
        """Create the exact SMOTE + Voting Classifier that performed best"""
        # Base models for voting classifier
        models = {
            'xgb': XGBClassifier(
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
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'et': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lgb'] = LGBMClassifier(
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
        
        # Create voting classifier
        voting_estimators = [(name, model) for name, model in models.items()]
        
        voting_classifier = VotingClassifier(
            estimators=voting_estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_classifier
    
    def prepare_data_pipeline(self, X, y):
        """
        Prepare data using the exact pipeline from best experiment:
        1. Label encoding
        2. Scaling
        3. SMOTE balancing
        4. RFECV feature selection
        """
        print("🔧 Preparing data with SMOTE + RFECV pipeline...")
        
        # Step 1: Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Step 2: Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 3: Apply SMOTE balancing
        print("   Applying SMOTE balancing...")
        smote = SMOTE(random_state=self.random_state, k_neighbors=3)
        try:
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)
            print(f"   ✅ SMOTE: {len(X_scaled)} → {len(X_balanced)} samples")
        except Exception as e:
            print(f"   ⚠️ SMOTE failed: {e}, using original data")
            X_balanced, y_balanced = X_scaled, y_encoded
        
        # Step 4: RFECV feature selection
        print("   Applying RFECV feature selection...")
        estimator = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        self.feature_selector = RFECV(
            estimator=estimator, 
            cv=3, 
            scoring='balanced_accuracy', 
            n_jobs=-1
        )
        X_selected = self.feature_selector.fit_transform(X_balanced, y_balanced)
        print(f"   ✅ RFECV: {X_balanced.shape[1]} → {X_selected.shape[1]} features")
        
        return X_selected, y_balanced
    
    def apply_pipeline_to_new_data(self, X_new, y_new=None):
        """Apply the fitted pipeline to new data (for validation splits)"""
        # Scale
        X_scaled = self.scaler.transform(X_new)
        
        # Apply SMOTE only if y_new is provided (training data)
        if y_new is not None:
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            try:
                X_balanced, y_balanced = smote.fit_resample(X_scaled, y_new)
            except:
                X_balanced, y_balanced = X_scaled, y_new
        else:
            X_balanced, y_balanced = X_scaled, y_new
        
        # Feature selection
        if hasattr(self.feature_selector, 'transform'):
            X_selected = self.feature_selector.transform(X_balanced)
        else:
            X_selected = X_balanced
        
        return X_selected, y_balanced
    
    def nested_cross_validation(self, X, y, outer_cv=5, inner_cv=3):
        """
        Gold Standard: Nested CV for unbiased performance estimate
        """
        print(f"\n🔄 Running Nested Cross-Validation ({outer_cv} outer, {inner_cv} inner folds)...")
        
        outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        nested_scores = []
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
            print(f"   Outer fold {fold + 1}/{outer_cv}")
            
            # Split data
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Apply pipeline to training data
            scaler = RobustScaler()
            label_encoder = LabelEncoder()
            
            y_train_encoded = label_encoder.fit_transform(y_train_outer)
            y_test_encoded = label_encoder.transform(y_test_outer)
            
            X_train_scaled = scaler.fit_transform(X_train_outer)
            X_test_scaled = scaler.transform(X_test_outer)
            
            # SMOTE on training data only
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            try:
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
            except:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
            
            # Feature selection on training data only
            estimator = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            feature_selector = RFECV(estimator=estimator, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)
            X_train_selected = feature_selector.fit_transform(X_train_balanced, y_train_balanced)
            X_test_selected = feature_selector.transform(X_test_scaled)
            
            # Train voting classifier
            voting_classifier = self.create_smote_voting_classifier()
            voting_classifier.fit(X_train_selected, y_train_balanced)
            
            # Predict on test fold
            y_pred = voting_classifier.predict(X_test_selected)
            fold_score = balanced_accuracy_score(y_test_encoded, y_pred)
            nested_scores.append(fold_score)
            
            fold_details.append({
                'fold': fold + 1,
                'train_size': len(y_train_balanced),
                'test_size': len(y_test_encoded),
                'balanced_accuracy': fold_score,
                'selected_features': X_train_selected.shape[1]
            })
            
            print(f"      Fold {fold + 1} Score: {fold_score:.4f}")
        
        nested_mean = np.mean(nested_scores)
        nested_std = np.std(nested_scores)
        
        self.validation_results['nested_cv'] = {
            'scores': nested_scores,
            'mean': nested_mean,
            'std': nested_std,
            'details': fold_details
        }
        
        print(f"\n✅ Nested CV Results:")
        print(f"   Mean Balanced Accuracy: {nested_mean:.4f} (+/- {nested_std*2:.4f})")
        print(f"   Individual Folds: {[f'{s:.4f}' for s in nested_scores]}")
        
        return nested_scores
    
    def temporal_validation(self, X, y, patients_df):
        """
        Temporal validation: Train on early patients, test on later patients
        """
        print(f"\n🕒 Running Temporal Validation...")
        
        # Sort by patient ID (assuming temporal order)
        patient_ids = [int(pid) for pid in patients_df['id']]
        sorted_indices = np.argsort(patient_ids)
        
        # 80/20 split
        split_point = int(0.8 * len(sorted_indices))
        train_indices = sorted_indices[:split_point]
        test_indices = sorted_indices[split_point:]
        
        print(f"   Train: Patients {min(patient_ids[i] for i in train_indices)} - {max(patient_ids[i] for i in train_indices)}")
        print(f"   Test: Patients {min(patient_ids[i] for i in test_indices)} - {max(patient_ids[i] for i in test_indices)}")
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Apply full pipeline
        scaler = RobustScaler()
        label_encoder = LabelEncoder()
        
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # SMOTE
        smote = SMOTE(random_state=self.random_state, k_neighbors=3)
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
        except:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
        
        # Feature selection
        estimator = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        feature_selector = RFECV(estimator=estimator, cv=3, scoring='balanced_accuracy', n_jobs=-1)
        X_train_selected = feature_selector.fit_transform(X_train_balanced, y_train_balanced)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        # Train and evaluate
        voting_classifier = self.create_smote_voting_classifier()
        voting_classifier.fit(X_train_selected, y_train_balanced)
        y_pred = voting_classifier.predict(X_test_selected)
        
        temporal_score = balanced_accuracy_score(y_test_encoded, y_pred)
        
        self.validation_results['temporal'] = {
            'balanced_accuracy': temporal_score,
            'train_size': len(y_train_balanced),
            'test_size': len(y_test_encoded),
            'selected_features': X_train_selected.shape[1]
        }
        
        print(f"✅ Temporal Validation Score: {temporal_score:.4f}")
        return temporal_score
    
    def leave_one_condition_out(self, X, y, patients_df):
        """
        Leave-One-Condition-Out: Test generalization to unseen conditions
        """
        print(f"\n🏥 Running Leave-One-Condition-Out Validation...")
        
        conditions = patients_df['condition'].values
        unique_conditions = np.unique(conditions)
        
        loco_scores = []
        condition_results = []
        
        for condition in unique_conditions:
            if np.sum(conditions == condition) < 5:  # Skip conditions with too few samples
                print(f"   ⚠️ Skipping {condition} (only {np.sum(conditions == condition)} samples)")
                continue
                
            print(f"   Testing on condition: {condition}")
            
            test_mask = conditions == condition
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            if len(np.unique(y_train)) < 2:  # Need at least 2 classes for training
                print(f"      ⚠️ Insufficient classes in training data")
                continue
            
            # Apply pipeline
            scaler = RobustScaler()
            label_encoder = LabelEncoder()
            
            try:
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_test_encoded = label_encoder.transform(y_test)
            except:
                print(f"      ⚠️ Label encoding failed for {condition}")
                continue
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # SMOTE
            smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, len(y_train)//2))
            try:
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
            except:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
            
            # Feature selection
            estimator = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            feature_selector = RFECV(estimator=estimator, cv=3, scoring='balanced_accuracy', n_jobs=-1)
            try:
                X_train_selected = feature_selector.fit_transform(X_train_balanced, y_train_balanced)
                X_test_selected = feature_selector.transform(X_test_scaled)
            except:
                X_train_selected, X_test_selected = X_train_balanced, X_test_scaled
            
            # Train and evaluate
            voting_classifier = self.create_smote_voting_classifier()
            voting_classifier.fit(X_train_selected, y_train_balanced)
            y_pred = voting_classifier.predict(X_test_selected)
            
            condition_score = balanced_accuracy_score(y_test_encoded, y_pred)
            loco_scores.append(condition_score)
            
            condition_results.append({
                'condition': condition,
                'score': condition_score,
                'test_size': len(y_test),
                'train_size': len(y_train_balanced)
            })
            
            print(f"      Score: {condition_score:.4f}")
        
        if loco_scores:
            self.validation_results['loco'] = {
                'scores': loco_scores,
                'mean': np.mean(loco_scores),
                'std': np.std(loco_scores),
                'details': condition_results
            }
            
            print(f"\n✅ LOCO Results:")
            print(f"   Mean Balanced Accuracy: {np.mean(loco_scores):.4f} (+/- {np.std(loco_scores)*2:.4f})")
        
        return loco_scores
    
    def bootstrap_validation(self, X, y, n_bootstraps=50):
        """
        Bootstrap validation: Test stability across random samples
        """
        print(f"\n🎲 Running Bootstrap Validation ({n_bootstraps} iterations)...")
        
        bootstrap_scores = []
        
        for i in range(n_bootstraps):
            if (i + 1) % 10 == 0:
                print(f"   Bootstrap {i + 1}/{n_bootstraps}")
            
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            if len(oob_indices) < 5:  # Need sufficient out-of-bag samples
                continue
            
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]
            X_oob, y_oob = X[oob_indices], y[oob_indices]
            
            # Apply pipeline
            scaler = RobustScaler()
            label_encoder = LabelEncoder()
            
            try:
                y_bootstrap_encoded = label_encoder.fit_transform(y_bootstrap)
                y_oob_encoded = label_encoder.transform(y_oob)
            except:
                continue
            
            X_bootstrap_scaled = scaler.fit_transform(X_bootstrap)
            X_oob_scaled = scaler.transform(X_oob)
            
            # SMOTE
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            try:
                X_bootstrap_balanced, y_bootstrap_balanced = smote.fit_resample(X_bootstrap_scaled, y_bootstrap_encoded)
            except:
                X_bootstrap_balanced, y_bootstrap_balanced = X_bootstrap_scaled, y_bootstrap_encoded
            
            # Feature selection
            estimator = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            feature_selector = RFECV(estimator=estimator, cv=3, scoring='balanced_accuracy', n_jobs=-1)
            try:
                X_bootstrap_selected = feature_selector.fit_transform(X_bootstrap_balanced, y_bootstrap_balanced)
                X_oob_selected = feature_selector.transform(X_oob_scaled)
            except:
                X_bootstrap_selected, X_oob_selected = X_bootstrap_balanced, X_oob_scaled
            
            # Train and evaluate
            voting_classifier = self.create_smote_voting_classifier()
            voting_classifier.fit(X_bootstrap_selected, y_bootstrap_balanced)
            y_pred = voting_classifier.predict(X_oob_selected)
            
            bootstrap_score = balanced_accuracy_score(y_oob_encoded, y_pred)
            bootstrap_scores.append(bootstrap_score)
        
        if bootstrap_scores:
            self.validation_results['bootstrap'] = {
                'scores': bootstrap_scores,
                'mean': np.mean(bootstrap_scores),
                'std': np.std(bootstrap_scores),
                'ci_lower': np.percentile(bootstrap_scores, 2.5),
                'ci_upper': np.percentile(bootstrap_scores, 97.5)
            }
            
            print(f"\n✅ Bootstrap Results:")
            print(f"   Mean: {np.mean(bootstrap_scores):.4f}")
            print(f"   95% CI: [{np.percentile(bootstrap_scores, 2.5):.4f}, {np.percentile(bootstrap_scores, 97.5):.4f}]")
        
        return bootstrap_scores
    
    def learning_curve_analysis(self, X, y):
        """
        Learning curves: Detect overfitting patterns
        """
        print(f"\n📈 Analyzing Learning Curves...")
        
        # Use simple model for learning curve (faster)
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'))
        ])
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_scores, val_scores = learning_curve(
            pipeline, X, y,
            train_sizes=train_sizes,
            cv=3,
            scoring='balanced_accuracy',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        self.validation_results['learning_curve'] = {
            'train_sizes': train_sizes,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std
        }
        
        # Plot
        plt.figure(figsize=(10, 6))
        train_sizes_abs = train_sizes * len(X)
        
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Balanced Accuracy')
        plt.title('Learning Curves: SMOTE + Voting Classifier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        final_gap = train_mean[-1] - val_mean[-1]
        
        print(f"✅ Learning Curve Analysis:")
        print(f"   Final Training Score: {train_mean[-1]:.4f}")
        print(f"   Final Validation Score: {val_mean[-1]:.4f}")
        print(f"   Train-Val Gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print("   ⚠️ Large gap suggests potential overfitting")
        elif final_gap > 0.05:
            print("   ⚠️ Moderate gap, monitor for overfitting")
        else:
            print("   ✅ Small gap suggests good generalization")
        
        return final_gap
    
    def comprehensive_validation_report(self):
        """
        Generate comprehensive validation report
        """
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        if not self.validation_results:
            print("❌ No validation results available. Run validations first.")
            return
        
        print(f"{'Validation Method':<25} {'Score':<15} {'Status':<30}")
        print("-" * 70)
        
        # Nested CV (most important)
        if 'nested_cv' in self.validation_results:
            score = self.validation_results['nested_cv']['mean']
            std = self.validation_results['nested_cv']['std']
            if score > 0.85:
                status = "✅ Excellent (>85%)"
            elif score > 0.75:
                status = "✅ Good (>75%)"
            elif score > 0.65:
                status = "⚠️ Moderate (>65%)"
            else:
                status = "❌ Poor (<65%)"
            print(f"{'Nested CV':<25} {score:.4f} ± {std:.3f} {status:<30}")
        
        # Temporal validation
        if 'temporal' in self.validation_results:
            score = self.validation_results['temporal']['balanced_accuracy']
            if score > 0.80:
                status = "✅ Robust temporal generalization"
            elif score > 0.70:
                status = "⚠️ Moderate temporal drop"
            else:
                status = "❌ Poor temporal generalization"
            print(f"{'Temporal':<25} {score:.4f}        {status:<30}")
        
        # LOCO validation
        if 'loco' in self.validation_results:
            score = self.validation_results['loco']['mean']
            std = self.validation_results['loco']['std']
            if score > 0.70:
                status = "✅ Good cross-condition performance"
            elif score > 0.60:
                status = "⚠️ Moderate cross-condition performance"
            else:
                status = "❌ Poor cross-condition performance"
            print(f"{'Leave-One-Condition-Out':<25} {score:.4f} ± {std:.3f} {status:<30}")
        
        # Bootstrap validation
        if 'bootstrap' in self.validation_results:
            score = self.validation_results['bootstrap']['mean']
            ci_width = (self.validation_results['bootstrap']['ci_upper'] - 
                       self.validation_results['bootstrap']['ci_lower'])
            if ci_width < 0.1:
                status = "✅ Stable performance"
            elif ci_width < 0.2:
                status = "⚠️ Moderate variability"
            else:
                status = "❌ High variability"
            print(f"{'Bootstrap':<25} {score:.4f}        {status:<30}")
        
        # Overall assessment
        print("\n" + "="*60)
        print("📊 OVERALL ASSESSMENT")
        print("="*60)
        
        red_flags = []
        
        # Check for concerning patterns
        if 'nested_cv' in self.validation_results:
            nested_score = self.validation_results['nested_cv']['mean']
            if nested_score < 0.8:  # Significantly lower than 98%
                red_flags.append(f"Nested CV ({nested_score:.3f}) much lower than original 98%")
        
        if 'temporal' in self.validation_results:
            temporal_score = self.validation_results['temporal']['balanced_accuracy']
            if 'nested_cv' in self.validation_results:
                temporal_drop = self.validation_results['nested_cv']['mean'] - temporal_score
                if temporal_drop > 0.15:
                    red_flags.append(f"Large temporal performance drop ({temporal_drop:.3f})")
        
        if 'bootstrap' in self.validation_results:
            ci_width = (self.validation_results['bootstrap']['ci_upper'] - 
                       self.validation_results['bootstrap']['ci_lower'])
            if ci_width > 0.2:
                red_flags.append(f"High bootstrap variability (CI width: {ci_width:.3f})")
        
        if red_flags:
            print("🚨 POTENTIAL CONCERNS:")
            for flag in red_flags:
                print(f"   ⚠️ {flag}")
            print("\n💡 RECOMMENDATION: Results may not be as robust as initial 98% suggests")
        else:
            print("✅ NO MAJOR CONCERNS DETECTED")
            print("   Your SMOTE + Voting Classifier results appear robust!")
            print("   The high performance is likely genuine and generalizable.")
        
        return len(red_flags) == 0

def main():
    """
    Main validation pipeline for SMOTE + Voting Classifier
    """
    print("🔍 SMOTE + Voting Classifier Overfitting Validation")
    print("="*60)
    
    # Load data using existing pipeline
    data_path = "/Users/kiranshay/projects/Parkinsons-ET/pads-dataset/preprocessed"  # Update path
    
    loader = PADSDataLoader(data_path, use_preprocessed=True)
    feature_extractor = FeatureExtractor()
    
    print("📂 Loading PADS dataset...")
    patients_df = loader.load_patient_data()
    
    if len(patients_df) == 0:
        print("❌ Error loading patient data")
        return
    
    print(f"✅ Loaded {len(patients_df)} patients")
    
    # Extract features (same as your enhanced_models.py)
    print("🔧 Extracting features...")
    features_list = []
    labels = []
    key_tasks = ["Relaxed1", "StretchHold", "HoldWeight", "TouchNose", "CrossArms"]
    
    for idx, patient in patients_df.iterrows():
        patient_id = int(patient['id'])
        condition = patient['condition']
        
        patient_features = extract_patient_metadata_features(patient)
        preprocessed_data = loader.load_preprocessed_data(patient_id)
        
        if preprocessed_data is not None:
            for task in key_tasks:
                task_data = loader.extract_task_data(preprocessed_data, task)
                if task_data is not None and task_data.shape[1] > 0:
                    for channel_idx in range(min(12, task_data.shape[1])):
                        signal = task_data[:, channel_idx]
                        if len(signal) > 50:
                            signal = signal[50:]  # Skip first 0.5 seconds
                        if len(signal) > 10:
                            features = feature_extractor.extract_all_features(signal)
                            if features:
                                patient_features.extend(list(features.values()))
            
            if len(patient_features) > 14:  # More than just metadata
                features_list.append(patient_features)
                labels.append(condition)
        
        if (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1}/{len(patients_df)} patients...")
    
    X_features = np.array(features_list)
    y = np.array(labels)
    
    print(f"✅ Feature extraction complete:")
    print(f"   Shape: {X_features.shape}")
    print(f"   Classes: {np.unique(y, return_counts=True)}")
    
    # Handle missing values
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Initialize validator
    validator = SMOTEVotingValidator(random_state=42)
    
    print(f"\n🚀 Starting comprehensive validation...")
    print("⏱️  Expected time: 10-15 minutes")
    
    # Run all validation methods
    try:
        # 1. Nested Cross-Validation (Most Important)
        nested_scores = validator.nested_cross_validation(X_features, y, outer_cv=5, inner_cv=3)
        
        # 2. Temporal Validation
        temporal_score = validator.temporal_validation(X_features, y, patients_df)
        
        # 3. Leave-One-Condition-Out
        loco_scores = validator.leave_one_condition_out(X_features, y, patients_df)
        
        # 4. Bootstrap Validation
        bootstrap_scores = validator.bootstrap_validation(X_features, y, n_bootstraps=50)
        
        # 5. Learning Curve Analysis
        learning_gap = validator.learning_curve_analysis(X_features, y)
        
        # Generate comprehensive report
        is_robust = validator.comprehensive_validation_report()
        
        # Final recommendation
        print(f"\n" + "="*80)
        print("🎯 FINAL RECOMMENDATION")
        print("="*80)
        
        if is_robust:
            print("🎉 CONGRATULATIONS!")
            print("   Your 98.18% SMOTE + Voting Classifier results appear to be GENUINE!")
            print("   The model shows strong generalization across multiple validation methods.")
            print("\n📝 Next Steps:")
            print("   ✅ Consider submitting to a top-tier journal")
            print("   ✅ Explore clinical validation studies")
            print("   ✅ Investigate deployment for real-world screening")
        else:
            print("⚠️  CAUTION RECOMMENDED")
            print("   While performance is good, some validation concerns were detected.")
            print("   The original 98% may be optimistic due to overfitting.")
            print("\n📝 Recommendations:")
            print("   🔍 Investigate data leakage or feature engineering issues")
            print("   📊 Consider external dataset validation")
            print("   🎯 Focus on nested CV results for publication")
        
        # Save results summary
        results_summary = {
            'original_performance': 0.9818,  # Your best result
            'validation_results': validator.validation_results,
            'is_robust': is_robust,
            'recommendation': 'robust' if is_robust else 'cautious'
        }
        
        # Optionally save to file
        import json
        with open('validation_results.json', 'w') as f:
            json.dump({k: v for k, v in results_summary.items() if k != 'validation_results'}, 
                     f, indent=2, default=str)
        
        print(f"\n💾 Results saved to 'validation_results.json'")
        
        return validator
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    validator = main()