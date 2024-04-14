import wandb
import os
import torch
from datetime import datetime

class ExperimentLogger():
    verbose = False
    wandb_run_name = None
    wandb_run = None
    metrics_output_dir = "./analysis/"
    model_output_dir = "./model/"
    
    @classmethod
    def init_wandb(cls, logger_conf, hydra_cfg):
        """Expecting wand_group_id, project, entity, and cfg in dictionary"""
        if not logger_conf.record:
            print("Record is set to false, not logging to wandb")
            return None
        try:
            ExperimentLogger.wandb_run = wandb.init(group=hydra_cfg["command"]["task"],
                                        project=logger_conf.project,
                                        entity=logger_conf.entity,
                                        config=hydra_cfg)
        except:
            raise Exception("Missing values from hydra config, check your setup")
        
    @classmethod
    def log(cls, data_dict):
        """
        Logs to the remote logging session
        """
        if ExperimentLogger.wandb_run is None:
            print("Logging locally and NOT to wandb")
            print(data_dict)
            return
        try:
            ExperimentLogger.wandb_run.log(data_dict)
        except:
            raise Exception("You need to initialize a remote logger before you start logging") 
        if ExperimentLogger.verbose:
            print(data_dict)

    @classmethod
    def write_table_wandb(cls, dataFrame, fileName=None):
        """
        Converts the dataframe to a csv file and writes it to the wandb run directory
        """
        if ExperimentLogger.wandb_run is None:
            print("Saving locally and NOT to wandb")
            if fileName is None:
                fileName = f'results_table_{datetime.now()}.csv'
            dataFrame.to_csv(os.path.join(ExperimentLogger.metrics_output_dir, fileName))
            return
        if fileName is None:
            fileName = f'results_table_{cls.wandb_run.name}.csv'
        try:
            dataFrame.to_csv(os.path.join(wandb.run.dir, fileName))
        except:
            raise Exception("Error when writing table to wandb. Maybe you need to initialize a remote logger?") 
        
    @classmethod
    def save_model_wanb(cls, model, fileName=None):
        """
        Saves the model to the wandb run directory
        """
        if ExperimentLogger.wandb_run is None:
            print("Saving locally and NOT to wandb")
            if fileName is None:
                fileName = f'final_model_{datetime.now()}.pt'
            torch.save(model, os.path.join(ExperimentLogger.model_output_dir, fileName))
            return
        if fileName is None:
            fileName = f'final_model_{cls.wandb_run.name}.pt'
        try:
            torch.save(model, os.path.join(wandb.run.dir, fileName))
        except:
            raise Exception("Error when saving model to wandb. Maybe you need to initialize a remote logger?")

    @classmethod
    def save_df_to_json(cls, dataFrame, fileName=None):
        """
        Saves the dataframe to the wandb run directory
        """
        if ExperimentLogger.wandb_run is None:
            print("Logging locally and NOT to wandb")
            if fileName is None:
                fileName = f'results_{datetime.now()}.json'
            dataFrame.to_json(os.path.join(ExperimentLogger.metrics_output_dir, fileName), orient='records', indent=4)
            return
        if fileName is None:
            fileName = f'results_{cls.wandb_run.name}.json'
        try:
            dataFrame.to_json(os.path.join(wandb.run.dir, fileName), orient='records', indent=4)
        except:
            raise Exception("Error when saving model to wandb. Maybe you need to initialize a remote logger?")
        
    @classmethod
    def get_run_name(cls):
        if ExperimentLogger.wandb_run is None:
            return f"local_run_{datetime.now()}"
        return ExperimentLogger.wandb_run.name