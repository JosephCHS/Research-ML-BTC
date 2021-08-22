# F21MP_2020 - Research-ML-BTC - H00359322


# Introduction

As part of the course *F21MP_2020-2021, Masters Project and Dissertation* carried out at Heriot-Watt University for the MSc Data Science, the **Research-ML-BTC** project was carried out.
This project is open-source and versioned on Git.


It is avaible on :
- HWU Gitlab MACS (requires an HWU ID): [Gitlab](https://gitlab-student.macs.hw.ac.uk/jmmc2000/f21mp_2020-2021)
- The Github of the author Joseph Chartois: [Github](https://github.com/JosephCHS/Research-ML-BTC)


# Documentation

The documentation is generated with the Sphinx framework with the Read The Doc template.
It is accessible via the file: `/documentation/build/html/index.html`

# Pre-requisites (linux)

1. Git clones the repository: 
- `git clone git@gitlab-student.macs.hw.ac.uk:jmmc2000/f21mp_2020-2021.git`
or
- `git clone git@github.com:JosephCHS/Research-ML-BTC.git`
2. Enter the directory 
3. Have Python 3 installed (default on recent OS like Ubuntu or Debian)
4. Install pip: `sudo apt install -y python3-pip`
5. Install virtual-env: `python3 -m pip install --user virtualenv`
6. Activate the virtual environment: `source .venv/bin/activate`
7. Install the prerequisites in the virtual environment: `pip install -r source/requirements.txt`


# Usage 

You can import modules from the project and use the classes and their methods.
In addition, documentation is available: `/documentation/build/html/index.html`
Examples of usage are written at the bottom of the files and commented to guide the user. 
These examples are also available bellow.


## Fetch
```
dataset = Dataset()  
dataset.create_dataset()
```

## Convert
```
convert = Arff()  
convert.generate_arff()  
convert.generate_arff_with_future()
```

## Chart

### Candlesticks
```
dataset = Dataset()  
dataframe_btc = dataset.get_btc_data()  
candlesticks = Candlesticks()  
candlesticks.display_candlesticks_chart(dataframe_btc)
```

### Confusion_matrix
```
ConfusionMatrix(True)
```

## Model
```
# Instantiate class  
machine_learning = MachineLearning()  
machine_learning.display_information()  
# Model sklearn  
models_sklearn = [  
    machine_learning.model_logistic_regression(),  
  machine_learning.model_svm(),  
]  
# Display models results sklearn  
for model in models_sklearn:  
    machine_learning.display_result(model)  
machine_learning.display_report_sklearn()  
# Model Keras  
machine_learning.model_lstm()  
machine_learning.model_cnn()  
# Model Pytorch  
machine_learning.model_bnn()
```
