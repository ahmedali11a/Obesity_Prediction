# Obesity AI - Machine Learning Classification Project

## 📋 Project Overview

This project implements a comprehensive machine learning solution for obesity classification using various algorithms. The system analyzes lifestyle and health data to predict obesity levels, providing insights for health professionals and individuals.

## 🎯 Problem Statement

Obesity is a significant global health concern. This project aims to:
- Classify individuals into different obesity categories based on lifestyle factors
- Provide accurate predictions using machine learning algorithms
- Help healthcare professionals assess obesity risk factors
- Enable individuals to understand their obesity classification

## 🚀 Features

### Core Functionality
- **Multi-class Classification**: Predicts 7 different obesity categories
- **Multiple ML Algorithms**: Implements 4 different machine learning models
- **Interactive Prediction**: Command-line interface for real-time predictions
- **Model Persistence**: Saves trained models for future use
- **Comprehensive Evaluation**: Detailed performance metrics and analysis

### Obesity Categories
1. **Insufficient_Weight** - Underweight individuals
2. **Normal_Weight** - Healthy weight range
3. **Overweight_Level_I** - Slightly overweight
4. **Overweight_Level_II** - Moderately overweight
5. **Obesity_Type_I** - Class I obesity
6. **Obesity_Type_II** - Class II obesity
7. **Obesity_Type_III** - Class III obesity (severe)

## 📊 Dataset Features

The model uses 16 input features to predict obesity levels:

### Physical Attributes
- **Age** - Individual's age in years
- **Height** - Height in meters
- **Weight** - Weight in kilograms

### Lifestyle Factors
- **FAVC** - Frequent consumption of high caloric food (yes/no)
- **FCVC** - Frequency of consumption of vegetables (1-3 scale)
- **NCP** - Number of main meals (1-4)
- **CAEC** - Consumption of food between meals (Never/Sometimes/Frequently/Always)
- **CH2O** - Daily consumption of water (liters)
- **FAF** - Physical activity frequency (0-3 scale)
- **TUE** - Time using technology devices (hours)

### Health & Family History
- **family_history_with_overweight** - Family history of overweight (yes/no)
- **SMOKE** - Smoking habit (yes/no)
- **SCC** - Calorie consumption monitoring (yes/no)
- **CALC** - Consumption of alcohol (Never/Sometimes/Frequently)

### Transportation
- **MTRANS** - Transportation used (Automobile/Bike/Public_Transportation/Walking/Motorbike)

## 🏗️ Architecture

### Data Pipeline
1. **Data Loading** - Training and test datasets
2. **Data Preprocessing** - Handling missing values, encoding categorical variables
3. **Feature Engineering** - BMI calculation, outlier handling
4. **Data Scaling** - Standardization of numerical features
5. **Model Training** - Multiple algorithm implementation
6. **Evaluation** - Performance metrics and comparison
7. **Model Persistence** - Saving trained models

### Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| **Logistic Regression** | 88.76% | 88.15% | 88.01% | 88.69% |
| **SVM (RBF Kernel)** | 93.82% | 91.94% | 91.92% | 92.32% |
| **Random Forest** | 100.00% | 97.63% | 97.86% | 97.70% |
| **Gradient Boosting** | 100.00% | 98.10% | 97.96% | 98.29% |

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see requirements.txt)

### Setup Instructions
1. Clone the repository:
```bash
git clone <repository-url>
cd Obesity_AI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Obesity_pred.ipynb
```

## 📁 Project Structure

```
Obesity_AI/
├── 📊 Data/
│   ├── train_dataset.csv          # Training dataset (1900 samples)
│   └── test_dataset.csv           # Test dataset (211 samples)
├── 🧠 Models/
│   ├── logistic_regression.pkl    # Trained Logistic Regression model
│   ├── svm.pkl                    # Trained SVM model
│   ├── random_forest.pkl          # Trained Random Forest model
│   ├── gradient_boosting.pkl      # Trained Gradient Boosting model
│   └── preprocessing.pkl          # Preprocessing pipeline
├── 📈 Visualizations/
│   ├── Distributions/             # Feature distribution plots
│   ├── Boxplots/                  # Before/after outlier handling
│   ├── Countplots/                # Categorical feature analysis
│   ├── Features_VS_Obesity_Level/ # Feature vs target analysis
│   └── Correlation/               # Correlation matrices
├── 📓 Notebooks/
│   └── Obesity_pred.ipynb         # Main analysis notebook
└── 📚 Documentation/
    ├── README.md                  # This file
    └── Obesity Prediction.pdf     # Detailed project report
```

## 🚀 Usage

### Training Models
1. Open `Obesity_pred.ipynb` in Jupyter
2. Run all cells to train all models
3. Models will be automatically saved to the `models/` directory

### Making Predictions
The notebook includes an interactive prediction interface:

```python
# Example usage
Age: 25
Height: 1.75
Weight: 80.0
family_history_with_overweight: yes
NCP: 3
CAEC: sometimes
CH2O: 2.5
FAF: 1.5
TUE: 0.5
CALC: sometimes
MTRANS: public_transportation

# Output: Predicted Obesity Category
```

### Using Saved Models
```python
import joblib

# Load preprocessing pipeline
preprocessing = joblib.load('models/preprocessing.pkl')

# Load trained model
model = joblib.load('models/gradient_boosting.pkl')

# Make predictions
predictions = model.predict(processed_data)
```

## 📊 Results & Analysis

### Model Performance
- **Best Overall Model**: Gradient Boosting (100% accuracy, 98.29% F1-score)
- **Most Reliable**: Random Forest (100% accuracy, 97.70% F1-score)
- **Good Balance**: SVM (93.82% accuracy, 92.32% F1-score)
- **Baseline**: Logistic Regression (88.76% accuracy, 88.69% F1-score)

### Key Insights
1. **Feature Importance**: Physical activity (FAF) and family history are strong predictors
2. **Data Quality**: Comprehensive preprocessing improved model performance significantly
3. **Model Selection**: Ensemble methods (Random Forest, Gradient Boosting) outperform single models
4. **Practical Application**: High accuracy makes the system suitable for clinical use

## 🔧 Technical Details

### Data Preprocessing
- **Missing Value Handling**: Imputation strategies for numerical and categorical features
- **Outlier Detection**: IQR method for identifying and handling extreme values
- **Feature Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Feature Engineering**: BMI calculation from height and weight

### Model Training
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Class Balance**: Handling imbalanced classes in obesity categories
- **Model Persistence**: Joblib for saving/loading trained models

## 📈 Visualization Features

The project includes comprehensive visualizations:
- **Distribution Plots**: Understanding feature distributions
- **Correlation Analysis**: Feature relationships and multicollinearity
- **Boxplots**: Outlier detection and handling
- **Countplots**: Categorical feature analysis
- **Feature vs Target**: Understanding prediction patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ahmed ali** 

## 🙏 Acknowledgments

- Dataset providers and contributors
- Scikit-learn development team
- Open-source community for tools and libraries
- Healthcare professionals for domain expertise

## 📞 Contact


- **Email**: ahmedatiea30@gmail.com

## 🔮 Future Enhancements

- **Web Application**: User-friendly web interface
- **API Development**: RESTful API for integration
- **Real-time Monitoring**: Continuous model updates
- **Mobile App**: Cross-platform mobile application
- **Advanced Analytics**: Deep learning approaches
- **Clinical Integration**: Healthcare system integration

---

⭐ **Star this repository if you find it helpful!**

*This project demonstrates the power of machine learning in healthcare applications and provides a solid foundation for obesity risk assessment.*
