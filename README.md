# PADS Dataset Baseline Models

A comprehensive machine learning baseline framework for the **Parkinson's Disease Smartwatch (PADS) dataset**, providing standard comparisons across traditional ML algorithms and neural networks for movement disorder classification.

## 🎯 Overview

This project implements baseline models for classifying Parkinson's disease and other movement disorders using smartwatch sensor data from the PADS dataset. The framework extracts comprehensive features from accelerometer and gyroscope signals and evaluates performance across multiple machine learning approaches.

## 📊 Dataset

**PADS (Parkinson's Disease Smartwatch Dataset)**
- **Source**: [PhysioNet](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)
- **Participants**: 469 individuals across 3 groups:
  - Parkinson's disease patients
  - Differential diagnoses (essential tremor, atypical Parkinsonism, etc.)
  - Healthy controls
- **Data Collection**: Bilateral smartwatch recordings (Apple Watch Series 4)
- **Tasks**: 11 neurologist-designed movement tasks (10-20 seconds each)
- **Sensors**: 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope) at 100 Hz

## 🏗️ Architecture

### Core Components

1. **`PADSDataLoader`**: Handles dataset loading and time series extraction
2. **`FeatureExtractor`**: Comprehensive signal feature extraction
3. **`PADSBaselineModels`**: Multiple ML model evaluation framework

### Feature Engineering

**Time Domain Features:**
- Statistical measures (mean, std, min, max, range, RMS)
- Distribution characteristics (skewness, kurtosis)
- Signal dynamics (zero crossings, peak count)

**Frequency Domain Features:**
- FFT-based spectral analysis
- Dominant frequency, spectral centroid
- Spectral energy and entropy

**Wavelet Features:**
- Multi-level wavelet decomposition (Daubechies-4)
- Energy and standard deviation at each level

**Patient Metadata Features:**
- Demographics (age, height, weight, gender)
- Clinical information (handedness, family history)
- Alcohol effect on tremor symptoms

## 🤖 Baseline Models

### Traditional Machine Learning
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting framework
- **Support Vector Machine**: RBF kernel classification
- **Logistic Regression**: Linear probabilistic classifier
- **k-Nearest Neighbors**: Instance-based learning
- **AdaBoost**: Adaptive boosting ensemble
- **Multi-Layer Perceptron**: Feedforward neural network
- **Voting Classifier**: Ensemble meta-classifier

### Deep Learning Models
- **LSTM**: Sequential pattern recognition
- **1D CNN**: Local temporal feature extraction
- **GRU**: Lightweight recurrent architecture

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn xgboost tensorflow pandas numpy scipy pywavelets matplotlib seaborn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kiranshay/Parkinsons-ET
cd Parkinsons-ET
```

2. **Download the PADS dataset**
   - Visit [PhysioNet PADS dataset](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)
   - Download and extract to your project directory

3. **Update the dataset path**
   ```python
   # In baseline_models.py, line ~420
   data_path = "/path/to/your/pads-dataset"  # Update this path
   ```

### Usage

**Basic execution:**
```bash
python baseline_models.py
```

**Expected output:**
- Dataset loading and validation
- Feature extraction progress
- Model evaluation results
- Performance comparison plots
- Results summary table

## 📈 Evaluation Metrics

- **Accuracy**: Overall classification performance
- **F1-Score**: Weighted average across classes
- **Cross-Validation**: 5-fold stratified validation
- **Classification Reports**: Per-class precision/recall
- **Confusion Matrices**: Detailed error analysis

## 🎛️ Configuration

### Customizing Feature Extraction

```python
# Modify in FeatureExtractor class
def extract_all_features(self, signal, sampling_rate=100):
    features = {}
    features.update(self.time_domain_features(signal))
    features.update(self.frequency_domain_features(signal, sampling_rate))
    features.update(self.wavelet_features(signal))
    return features
```

### Adding New Models

```python
# In PADSBaselineModels.initialize_models()
self.models['Your Model'] = YourModelClass(parameters)
```

### Adjusting Neural Network Architecture

```python
# Example: Modify LSTM layers
def create_lstm_model(self, input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),  # Increased units
        Dropout(0.3),  # Increased dropout
        LSTM(64),
        # ... rest of architecture
    ])
```

## 📁 Project Structure

```
Parkinsons-ET/
└── baseline_models/            # scripts folder
    ├── baseline_models.py      # scripts (updated as need be)
    ├── baseline_models_v2.py
├── README.md                   # This file
├── requirements.txt            # Dependencies
└── pads-dataset/              # Dataset directory
    ├── movement/              # Movement task data
    ├── patients/              # Patient metadata
    ├── questionnaire/         # Survey responses
    └── preprocessed/          # Pre-computed features
```

## 🔧 Troubleshooting

### Common Issues

**"Loaded data for 0 patients"**
- Check dataset path in `data_path` variable
- Verify dataset directory structure
- Ensure `patients/` subdirectory exists with `patient_*.json` files

**TensorFlow/Keras Import Errors**
```bash
# Try different TensorFlow versions
pip install tensorflow==2.13.0
# For Apple Silicon:
pip install tensorflow-macos tensorflow-metal
```

**Memory Issues with Large Dataset**
- Reduce batch size in neural network training
- Use feature selection to reduce dimensionality
- Process data in chunks for very large datasets

## 📊 Expected Results

The baseline framework provides comprehensive performance benchmarks across multiple algorithms. Typical results show:

- **Traditional ML**: Random Forest and XGBoost often perform well
- **Neural Networks**: LSTM can capture temporal dependencies effectively
- **Feature Importance**: Frequency domain features often most discriminative
- **Class Imbalance**: Framework handles imbalanced classes with stratified sampling

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Advanced Feature Engineering**: Spectral features, non-linear dynamics
2. **Deep Learning Architectures**: Transformers, attention mechanisms
3. **Ensemble Methods**: Stacking, advanced voting strategies
4. **Preprocessing**: Signal filtering, normalization techniques
5. **Evaluation Metrics**: Disease-specific metrics, clinical relevance

## 📝 Citation

If you use this baseline framework in your research, please cite:

**Original PADS Dataset:**
```bibtex
@article{varghese2024machine,
  title={Machine Learning in the Parkinson's disease smartwatch (PADS) dataset},
  author={Varghese, Julian and Brenner, Alexander and Fujarski, Michael and van Alen, Catharina M and Plagwitz, Lucas and Warnecke, Tobias},
  journal={npj Parkinson's Disease},
  volume={10},
  number={1},
  pages={9},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [PADS Dataset on PhysioNet](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)
- [Original Paper](https://www.nature.com/articles/s41531-024-00632-6)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

---

**Note**: This framework provides baseline comparisons for research purposes. Clinical applications require additional validation and regulatory approval.
