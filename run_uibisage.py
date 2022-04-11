import pickle
import logging
import os
from logging import getLogger
import json
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.model.sequential_recommender import UIBiSage
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, InputType, set_color
import argparse

def run_recbole(model_name, data_name):
    config = Config(model=UIBiSage, dataset=data_name, 
                    config_file_list=['{dataname}.yaml'.format(dataname=data_name)])
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    # Train/Valid/Test Split
    dataloader_file = './saved/{data_name}/{data_name}-for-{model_name}-dataloader.pth'.format(data_name=data_name, model_name=model_name)
    is_dataloader = os.path.isfile(dataloader_file)
    print(is_dataloader)
    if is_dataloader:
        logger.info("load data")
        train_data, valid_data, test_data  = load_split_dataloaders(config)
    else:
        # # dataset filtering
        dataset = create_dataset(config)
        dataset.save()
        train_data, valid_data, test_data = data_preparation(config, dataset)
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))
        
    # Model load
    # model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model = UIBiSage(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = Trainer(config, model)
    
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    return best_valid_result, test_result

if __name__ == "__main__":
    # DATA_LIST = ['tmall-buy','steam']
    # MODEL_LIST = ['SASRec', 'Caser', 'GRU4Rec']
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, help='name of the datasets', required=True)
    parser.add_argument('--model_name', type=str, help='name of the models', required=True)
    print("Recbole Start!")
    args = parser.parse_args()
    print(args.__dict__)
    
    best_valid_result, test_result = run_recbole(args.model_name, args.data_name)
    
    with open('./saved/{}/{}-valid-result.json'.format(args.data_name, args.model_name), 'w') as f:
        json.dump(dict(best_valid_result), f)
    f.close()
    
    with open('./saved/{}/{}-test-result.json'.format(args.data_name, args.model_name), 'w') as f:
        json.dump(dict(test_result), f)       
    f.close()