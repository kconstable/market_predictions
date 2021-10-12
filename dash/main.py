import visuals
import preprocess
import json
from datetime import datetime

# constants
UPDATE = True
ROOT_PATH = '/home/ken/PycharmProjects/price_predictions/'
WEIGHTS = {'lstm': 0.448420, 'fbp': 0.446634, 'last_close': 0.104946}


def get_av_key(path):
    """
    Retrieves the alpha advantage API key from the config file
    :param path: path to the av key
    :return: key (str)
    """
    with open(path + 'av_key.txt') as f:
        key = f.read().strip()

    return key


def get_config(path):
    """
    Retrieves config files for the main/data processes
    :param path: path to the config file
    :return: A dict of the config file
    """
    with open(path, 'r') as fp:
        config = json.load(fp)
    return config


# Control Flow for the application updates
# This code is run everyday at 4:45pm to get the latest prices and updates the df_hist file for
# each stock/crypto in the config file
# all config settings are contained in config files
#   config_main => stock/crypto configuration.  Contains paths to feature lists, df_hist files, and data config files
#   config_stock/crypto => contains the data config requirements for each stock/crypto listed in config_main
if __name__ == '__main__':
    # get config files
    KEY = get_av_key(ROOT_PATH + 'config/')
    config_main = get_config(ROOT_PATH + 'config/config_main.json')

    if UPDATE:
        # start the update
        for c in config_main:
            # get file paths from config_main
            model_path = ROOT_PATH + 'models/' + c['model']['modelname']
            feature_path = ROOT_PATH + 'data/' + c['model']['features']
            df_hist_path = ROOT_PATH + 'data/' + c['model']['df_hist']

            # extract info from config_main
            stock_type = c['stock']['type']
            stock = c['stock']['name']
            transform = c['stock']['transform']
            shift = c['stock']['shift']
            n_steps = c['model']['n_steps']
            n_predict = c['model']['n_predict']

            config_data_path = ROOT_PATH + f"config/{c['stock']['data']}"
            config_data = get_config(config_data_path)

            if (stock_type == 'stock' and datetime.today().weekday() <= 5) or (stock_type == 'crypto'):
                # update stocks mon-friday, crypto everyday
                print(f"{stock}:{stock_type}|{transform}|{config_data['Commodities']}")
                print('#' * 60)

                # update predictions for the next period
                df_hist_new = preprocess.update_predictions(stock, stock_type, config_data, transform, shift,
                                                            feature_path, n_steps, n_predict, model_path, df_hist_path,
                                                            KEY, WEIGHTS)
                print('\nComplete', '=' * 52, '\n\n')

                # plot the updated prices and predictions
                fig = visuals.plot_actual_ensemble(stock, df_hist_new, 'ens')
                fig.show()
            else:
                print('Weekend/Holiday')
                print('=' * 60)
                print(f'=>{stock} prices not updated\n\n')
    else:
        print("Process Run: Data not updated")
