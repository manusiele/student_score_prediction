# Student Score Prediction using Feedforward Neural Network

A deep learning project that predicts a student's future academic performance based on 8 subject marks using a Feedforward Neural Network (FNN) built with PyTorch.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a machine learning solution to predict student academic performance. The model takes 8 subject marks as input (in percentage format, 0-100%) and predicts the student's future score, helping educators identify at-risk students early.

### Key Objectives
- Predict future academic performance based on current subject marks
- Provide early warning system for at-risk students
- Demonstrate practical application of neural networks in education

## ✨ Features

- **Interactive Input**: User-friendly prompts for entering 8 subject marks
- **Percentage-based**: Accepts marks in familiar 0-100% format
- **Real-time Prediction**: Instant prediction of future performance
- **Visual Analytics**: Training curves, prediction plots, and residual analysis
- **Model Persistence**: Save and load trained models
- **Early Stopping**: Prevents overfitting during training
- **Comprehensive Evaluation**: MAE, RMSE, and R² metrics

## 📊 Dataset

**Source**: [Student Performance Dataset](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set) from Kaggle

**Description**: 
- Contains student grades and demographic information
- 649 samples with 33 features
- Target variable: G3 (final grade, 0-20 scale)

**Input Features** (8 subjects):
1. Math
2. English
3. Science
4. History
5. Geography
6. Physics
7. Chemistry
8. Biology

*Note: The actual dataset uses G1, G2, and other numeric features as proxies for subject marks.*

## 🏗️ Model Architecture

### Feedforward Neural Network (FNN)

```
Input Layer (8 features)
    ↓
Hidden Layer 1: 64 neurons + BatchNorm + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2: 32 neurons + BatchNorm + ReLU + Dropout(0.2)
    ↓
Hidden Layer 3: 16 neurons + ReLU
    ↓
Output Layer: 1 neuron (predicted score)
```

**Total Parameters**: 3,393

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | MSE (Mean Squared Error) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Batch Size | 32 |
| Max Epochs | 200 |
| Early Stopping | Patience=20 |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Train/Val/Test Split | 70% / 15% / 15% |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- Kaggle account (for dataset access)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/manusiele/student_score_prediction.git
cd student_score_prediction
```

2. **Install dependencies**
```bash
pip install kaggle pandas numpy matplotlib seaborn scikit-learn torch torchvision
```

3. **Get Kaggle API credentials**
   - Go to [Kaggle](https://www.kaggle.com) → Account → Create New API Token
   - Download `kaggle.json`

4. **For Google Colab**
   - Upload the notebook to Colab
   - Upload `kaggle.json` to Colab's file browser
   - Run all cells

5. **For Local Environment**
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
   - Run Jupyter: `jupyter notebook student_score_prediction.ipynb`

## 💻 Usage

### Training the Model

1. Open the notebook in Google Colab or Jupyter
2. Run cells sequentially from top to bottom
3. The notebook will:
   - Download the dataset automatically
   - Preprocess the data
   - Train the model with early stopping
   - Display training curves and metrics

### Making Predictions

After training, run the prediction cell and enter marks when prompted:

```
Enter marks for 8 subjects (0-100 percentage):

Math        : 75
English     : 82
Science     : 68
History     : 71
Geography   : 79
Physics     : 73
Chemistry   : 65
Biology     : 80
```

**Output:**
```
=======================================================
       STUDENT FUTURE SCORE PREDICTION
=======================================================

Input Subject Marks:
  Math        :  75.0%
  English     :  82.0%
  Science     :  68.0%
  History     :  71.0%
  Geography   :  79.0%
  Physics     :  73.0%
  Chemistry   :  65.0%
  Biology     :  80.0%
=======================================================
  PREDICTED FUTURE SCORE: 74.5%
  Status: PASS ✓
=======================================================
```

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | ~1.5 points |
| RMSE (Root Mean Squared Error) | ~2.0 points |
| R² Score | ~0.85 |

### Training Results
- Training converged at epoch 60 (early stopping)
- Best validation MSE: ~2.0
- No significant overfitting observed

### Visualizations

The notebook includes:
1. **Training Curves**: Loss vs. Epochs for train and validation sets
2. **Prediction Scatter Plot**: Actual vs. Predicted scores
3. **Residual Distribution**: Error analysis
4. **Feature Importance**: Permutation-based importance ranking

## 📁 Project Structure

```
student_score_prediction/
│
├── student_score_prediction.ipynb   # Main notebook
├── README.md                         # Project documentation
├── .gitignore                        # Git ignore file
│
├── models/                           # Saved models (generated)
│   ├── student_fnn.pth              # Trained model weights
│   └── scaler.pkl                   # Feature scaler
│
└── data/                            # Dataset (auto-downloaded)
    └── student-por.csv              # Student performance data
```

## 🛠️ Technologies Used

### Core Libraries
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Preprocessing and metrics

### Visualization
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualizations

### Development
- **Jupyter Notebook**: Interactive development
- **Google Colab**: Cloud-based execution
- **Kaggle API**: Dataset access

## 🔧 Customization

### Modifying the Model

To change the architecture, edit the `StudentFNN` class:

```python
class StudentFNN(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # Change layer sizes here
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Add more layers...
        )
```

### Adjusting Hyperparameters

Modify training parameters in the training cell:

```python
EPOCHS = 200          # Maximum epochs
PATIENCE = 20         # Early stopping patience
BATCH_SIZE = 32       # Batch size
LEARNING_RATE = 1e-3  # Learning rate
```

## 📊 Performance Optimization

### Tips for Better Results

1. **More Data**: Collect more student records
2. **Feature Engineering**: Add more relevant features
3. **Hyperparameter Tuning**: Use grid search or Optuna
4. **Ensemble Methods**: Combine multiple models
5. **Cross-Validation**: Use k-fold validation

## 🐛 Troubleshooting

### Common Issues

**Issue**: `kaggle.json not found`
- **Solution**: Upload kaggle.json to `/content/` in Colab or place in `~/.kaggle/` locally

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
- **Solution**: Run `pip install pandas numpy torch scikit-learn matplotlib seaborn`

**Issue**: CSV parsing error (single column)
- **Solution**: The notebook includes automatic CSV parsing fixes. Re-run the data loading cell.

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use CPU by setting `DEVICE = 'cpu'`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Manus** - *Initial work* - [manusiele](https://github.com/manusiele)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)
- Inspired by educational data mining research
- Built with PyTorch and Google Colab

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the repository owner.

---

**Note**: This project is for educational purposes. Predictions should be used as one of many factors in assessing student performance, not as the sole determinant.
