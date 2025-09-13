# Deep Learning Assignment - Object Classification

This repository contains a comprehensive deep learning assignment focused on object classification using various approaches including traditional computer vision features, pre-trained neural networks, and noise robustness analysis.

## 📊 Project Overview

This project implements and compares multiple machine learning approaches for object classification on a subset of the Caltech-256 dataset containing 90 object categories with 11,055 images total.

### Key Features
- Traditional computer vision feature extraction (HOG, LBP, Edge Detection)
- Deep learning feature extraction using pre-trained models (ResNet50, MobileNetV2, VGG16)
- Noise robustness analysis with Gaussian noise injection
- Comprehensive classifier comparison (Logistic Regression, SVM, Random Forest)
- Detailed performance evaluation with multiple metrics

## 🗂️ Repository Structure

DL_Assignment/
├── 256_ObjectCategories/ # Dataset folder with 90 object classes
│ ├── 001_ak47/
│ ├── 002_american-flag/
│ ├── ...
│ └── 253_faces-easy-101/
├── Outputs/ # Results and analysis outputs
│ ├── visuals/ # Generated visualizations
│ │ ├── samples/ # Sample processed images
│ │ └── classifiers_by_feature.png
│ └── label_map.csv # Class label mappings
├── second.ipynb # Main Jupyter notebook
└── README.md # This file


## 🚀 Getting Started

### Prerequisites
pip install torch torchvision scikit-learn scikit-image opencv-python
pip install matplotlib seaborn pandas numpy tqdm

### Dataset
The project uses a subset of the Caltech-256 dataset with 90 carefully selected object categories, including:
- Vehicles (airplanes, cars, fire trucks)
- Animals (bears, dolphins, elephants, zebras)
- Objects (guitars, laptops, umbrellas)
- And many more...

## 🔬 Methodology

### 1. Traditional Computer Vision Features
- **HOG (Histogram of Oriented Gradients)**: Captures shape and structure
- **LBP (Local Binary Patterns)**: Texture analysis
- **Edge Detection**: Canny, Prewitt, Sobel, Scharr, and Laplacian operators

### 2. Deep Learning Features
- **ResNet50**: Deep residual network for robust feature extraction
- **MobileNetV2**: Lightweight architecture for efficient processing
- **VGG16**: Classic deep architecture with strong feature representation

### 3. Classification Methods
- **Logistic Regression**: Linear classifier with multinomial approach
- **Support Vector Machine**: RBF kernel for non-linear classification
- **Random Forest**: Ensemble method with 300 trees

### 4. Noise Robustness Analysis
- Gaussian noise injection (μ=0, σ=0.1)
- Performance comparison between clean and noisy datasets
- Robustness evaluation across different architectures

## 📈 Key Results

### Deep Learning Performance (Clean Data)
| Model       | Classifier         | Accuracy | F1-Score | Kappa  |
|-------------|--------------------|----------|----------|--------|
| ResNet50    | Logistic Regression| 89.87%   | 88.24%   | 89.69% |
| ResNet50    | SVM                | 88.01%   | 86.08%   | 87.81% |
| MobileNetV2 | Logistic Regression| 87.97%   | 86.09%   | 87.76% |
| VGG16       | Logistic Regression| 86.21%   | 84.13%   | 85.97% |

### Traditional Features Performance
| Feature Type | Best Classifier | Accuracy | F1-Score |
|--------------|----------------|----------|----------|
| HOG | SVM (RBF) | 41.70% | 33.57% |
| Edge Detection | SVM (RBF) | ~28% | ~18% |
| LBP | SVM (RBF) | 23.02% | 12.48% |

### Noise Impact Analysis
The addition of Gaussian noise significantly impacts performance:
- ResNet50: ~20-25% accuracy drop
- MobileNetV2: ~20% accuracy drop
- VGG16: ~24% accuracy drop

## 🛠️ Usage

### Running the Main Analysis

Load and run the main notebook
jupyter notebook second.ipynb

### Key Functions
- `build_backbone(name)`: Initialize pre-trained models
- `deep_embeddings(loader, model)`: Extract deep features
- `evaluate_deep()`: Comprehensive evaluation with multiple classifiers
- `NoiseDataset`: Custom dataset wrapper for noise injection

### Feature Extraction
The notebook includes implementations for:
- Deep feature extraction using PyTorch models
- Traditional CV feature computation
- PCA dimensionality reduction (default: 256 dimensions)
- Train/test splitting with stratification

## 📊 Visualizations

The project generates several visualizations:
- Confusion matrices for top-performing classes
- ROC curves for multi-class classification  
- Feature extraction examples
- Noise impact comparisons
- Classifier performance comparisons

## 🔍 Analysis Insights

1. **Deep Learning Superiority**: Pre-trained models significantly outperform traditional features
2. **ResNet50 Leadership**: Consistently achieves the highest accuracy across classifiers
3. **Noise Vulnerability**: All models show substantial performance degradation with noise
4. **Feature Efficiency**: PCA reduction to 256 dimensions maintains good performance
5. **Classifier Consistency**: Logistic Regression and SVM show similar performance patterns

## 📝 Evaluation Metrics

The project uses comprehensive evaluation metrics:
- **Accuracy**: Overall classification correctness
- **Precision/Recall**: Class-wise performance analysis
- **F1-Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement measurement accounting for chance
- **Matthews Correlation Coefficient**: Balanced measure for multi-class problems

## 🤝 Contributing

This is an academic assignment repository. For suggestions or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is part of an academic assignment and is intended for educational purposes.

## 🔗 References

- Caltech-256 Object Category Dataset
https://www.kaggle.com/datasets/jessicali9530/caltech256
- PyTorch Pre-trained Models
- Scikit-learn Machine Learning Library
- Traditional Computer Vision Feature Extraction Methods

---

**Authors**: Adithya, Aniketh, Himanshu, Kavya   
**Course**: Deep Learning Assignment  
**Last Updated**: September 2025
