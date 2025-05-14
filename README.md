# Research-ML-BTC

A machine learning research project for Bitcoin price prediction using various ML/DL approaches including LSTM, CNN, and Bayesian Neural Networks.

## Requirements

- Python 3.12.3 or higher
- CUDA capable GPU (optional, for GPU acceleration)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JosephCHS/Research-ML-BTC.git 
cd Research-ML-BTC
```

2. Setup the environment:
```bash
make init
```
This will:
- Ensure correct Python version is installed (3.12.3)
- Create a virtual environment
- Install all required dependencies

## Usage

To run the complete analysis:
```bash
make run
```

To clean up generated files and virtual environment:
```bash
make clean
```

## Project Structure

- `source/` - Main source code
  - `model/` - Machine learning models implementation
  - `fetch/` - Data collection and preprocessing
  - `requirements.txt` - Python dependencies

## Models Implemented

1. Traditional ML:
   - Logistic Regression
   - Support Vector Machine (SVM)

2. Deep Learning:
   - Long Short-Term Memory (LSTM)
   - Convolutional Neural Network (CNN)
   - Bayesian Neural Network (BNN)

## Documentation

Generated documentation is available at `/documentation/build/html/index.html`

## Technical Details

- TensorFlow 2.x with CUDA support
- Python 3.12.3
- Scikit-learn for traditional ML models
- Yellowbrick for ML visualization
- TensorFlow Probability for Bayesian Neural Networks

## Example Usage

```python
from model.machine_learning import MachineLearning

# Initialize models
ml = MachineLearning()

# Train and evaluate traditional ML models
pred_lr = ml.model_logistic_regression()
pred_svm = ml.model_svm()

# Train and evaluate deep learning models
pred_lstm = ml.model_lstm()
pred_cnn = ml.model_cnn()
pred_bnn = ml.model_bnn()
```
