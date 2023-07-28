import pickle
from comet_ml import Experiment


# Assuming you have already defined your PyTorch model and data for training
# Initialize your CometML experiment


experiment = Experiment(api_key="5GQUeJbYop4ZaqBrtCQ12YaXh", project_name="unsupervised", workspace="vascoeti")
    

def log_metrics(experiment, num_recommendations, recommendation_time):
    # Log the number of recommendations made
    experiment.log_metric("num_recommendations", num_recommendations)
    # Log the time taken for the recommendation
    experiment.log_metric("recommendation_time", recommendation_time)

def log_model(experiment, model, model_name):
    with open(f"{model_name}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    experiment.log_asset(f"{model_name}.pkl", file_name=model_name + ".pkl")

def log_model(experiment, model, model_name):
    with open(f"{model_name}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    experiment.log_asset(f"{model_name}.pkl", file_name=model_name + ".pkl")
