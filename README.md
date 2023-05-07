# Development of simulated traffic driver agents
This project aims to develop a simulated traffic driver agent using a combination of imitation learning and reinforcement learning techniques. The simulated agent will be trained to navigate through a virtual city while obeying traffic laws and avoiding collisions with other vehicles and pedestrians.

# Project Overview
The project will be developed in several stages:

**Data Collection:** Collect data from real-world drivers using cameras mounted on their vehicles. This data will be used to train an imitation learning model to mimic the behavior of human drivers.

**Imitation Learning:** Train an imitation learning model using the collected data. The model will learn to predict the actions of human drivers based on input data such as vehicle speed, position, and surrounding objects.

**Reinforcement Learning:** Train a reinforcement learning model to navigate through a virtual city while obeying traffic laws and avoiding collisions. The model will be rewarded for following traffic laws, staying within speed limits, and avoiding collisions with other vehicles and pedestrians.

**Combination:** Combine the imitation learning and reinforcement learning models to create a simulated traffic driver agent. The imitation learning model will provide the initial driving behavior, while the reinforcement learning model will improve the behavior based on feedback from the virtual environment.

**Evaluation:** Evaluate the performance of the simulated traffic driver agent in various scenarios, including different weather conditions and traffic densities.

# Technologies Used
The project will be developed using the following technologies:

- Python programming language
- PyTorch libraries for machine learning
- Unity game engine for developing the virtual environment
- CARLA for traffic simulation

## Conclusion
The development of a simulated traffic driver agent using a combination of imitation learning and reinforcement learning techniques is an exciting project that has the potential to revolutionize the way we think about transportation. By training virtual agents to navigate through a virtual city while obeying traffic laws and avoiding collisions, we can create safer, more efficient, and more sustainable transportation systems.


Reinforcement Learning
==============================

Imitation Learning combined with Reinforcement Learning for the development of simulated traffic driver agents.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Setup Conda Environment
```
conda env create -f environment.yml
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Unit Testing
Run the following command to perform unit testing on individual components.
```
python -m unittest tests/test_actor.py
python -m unittest tests/test_critic.py
python -m unittest tests/test_actor_critic.py
```
