import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import pickle
import os
import glob

def get_result(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    predict = []
    target = []
    for item in data:
        predict.extend(item['predict'][item['select_atom'] == 1].cpu().numpy())
        target.extend(item['target'][item['select_atom'] == 1].cpu().numpy())
    return target, predict

def reg_metrics(target, predict):
    r2 = r2_score(target, predict)
    mae = mean_absolute_error(target, predict)
    mse = mean_squared_error(target, predict)
    rmse = math.sqrt(mse)
    return r2, mae, mse, rmse

def plot_metrics(target, predict):
    r2, mae, mse, rmse = reg_metrics(target, predict)
    plt.figure(figsize=(8, 6))
    plt.scatter(target, predict, color='blue', label='Actual vs Predicted')
    plt.plot([min(target), max(target)], [min(target), max(target)], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted\nMAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}'.format(mae, rmse, r2))
    plt.legend()
    plt.show()


def main(args):
    if args.mode == 'cv':
        target = 0
        all_predict = []
        for folder in os.listdir(args.path):
            folder_path = os.path.join(args.path, folder)
            if os.path.isdir(folder_path):
                pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
                filename = pkl_files[0]
                target, predict = get_result(filename)
                all_predict.append(predict)
                r2, mae, mse, rmse = reg_metrics(target, predict)
                print(f'metric of {filename}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        if all_predict:
            mean_predict = np.mean(np.vstack(all_predict), axis=0)
            plot_metrics(target, mean_predict)
            r2, mae, mse, rmse = reg_metrics(target, mean_predict)
            print(f'metric of mean\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

    else :
        pkl_files = glob.glob(os.path.join(args.path, "*.pkl"))
        filename = pkl_files[0]
        target, predict = get_result(filename)
        r2, mae, mse, rmse = reg_metrics(target, predict)
        print(f'metric of {filename}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        plot_metrics(target, predict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--path', type=str)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    main(args)