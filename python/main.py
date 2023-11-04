from utils.learning_loop import LearningAlgorithm
import os
import yaml

if __name__ == '__main__':
    config = yaml.safe_load(open('../config.yaml'))
    try:
        os.mkdir(config['log_dir'])
    except FileExistsError:
        print('Log directory already exist')

    model = LearningAlgorithm(config)
    model.run()
