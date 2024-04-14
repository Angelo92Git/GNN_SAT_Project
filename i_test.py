import torch
from torch.nn import BCELoss
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from tqdm import tqdm, trange
from itertools import product
import numpy as np
import argparse
import pandas as pd

from my_utils import formula_utils as F
from my_utils import graph_utils as f2g
import models as m

def main():
    parser = argparse.ArgumentParser(description="Test models and save results.")
    parser.add_argument("-d", "--difficulty", type=str, choices=["easy", "medium", "hard"], required=True, nargs='+', help="easy, medium, or hard, (combinations may also be provided)")
    parser.add_argument("-p", "--problem_type", type=str, required=True, nargs='+', help="3-sat, or sr, or both")
    parser.add_argument("-mp", "--model_params", type=str, required=True, nargs='+', help=".pt file in model params excluding '{m/d}_'")

    args = parser.parse_args()

    if type(args.difficulty) == list:
        test_difficulty_levels = args.difficulty
    else:
        test_difficulty_levels = [args.difficulty]
    
    if type(args.problem_type) == list:
        test_problem_types = args.problem_type
    else:
        test_problem_types = [args.problem_type]

    if type(args.model_params) == list:
        test_model_params = args.model_params
    else:
        test_model_params = [args.model_params]

    product_difficulty_problem_params = product(test_problem_types, test_difficulty_levels, test_model_params)
    test_splits = ["test"]
    satisfiability = ["sat", "unsat"]

    results = {"datetime":[], "model_parameters": [], "problem_types":[], "difficulty_levels":[], "accuracy":[], "precision":[], "recall":[], "f1":[]}
    for combo in product_difficulty_problem_params:
        test_problem_type = combo[0]
        test_difficulty_level = combo[1]
        test_model_param = combo[2]
        paths_to_testing_instances = F.get_paths_to_instances([test_difficulty_level], [test_problem_type], test_splits, satisfiability, "./dataset")
        testing_formulas = F.read_instances_from_paths(paths_to_testing_instances)
        if "VCGm" in test_model_param:
            testing_dataset = [f2g.convert_instance_to_VCG_bi_with_meta_node(formula) for formula in testing_formulas]
            include_meta_node = True
            label_key = "variable"
        elif "VCG" in test_model_param:
            testing_dataset = [f2g.convert_instance_to_VCG_bi(formula) for formula in testing_formulas]
            include_meta_node = False
            label_key = "variable"
        elif "LCGm" in test_model_param:
            testing_dataset = [f2g.convert_instance_to_LCG_with_meta_node(formula) for formula in testing_formulas]
            include_meta_node = True
            label_key = "literal"
        elif "LCG" in test_model_param:
            testing_dataset = [f2g.convert_instance_to_LCG(formula) for formula in testing_formulas]
            include_meta_node = False
            label_key = "literal"
        else:
            KeyError(f"Model representation {test_model_param} not recognized.")

        # Create data loaders
        num_testing_instances = len(testing_dataset)
        print(f"Number of testing instances: {num_testing_instances}")
        test_loader = DataLoader(testing_dataset, batch_size=20, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        params = test_model_param.split("_")
        param_dict = {"model": params[0], "representation": params[1], "latent_dim": int(params[4][2:]), "num_conv_layers": int(params[5][1:])}
        models_available = {"GCN": {"VCG":m.G4GCN_VCG, "LCG":m.G4GCN_LCG, "VCG":m.G4GCN_VCG, "LCG":m.G4GCN_LCG}}
        hidden_channels = param_dict["latent_dim"]
        num_conv_layers = param_dict["num_conv_layers"]
        model = models_available[param_dict["model"]][param_dict["representation"]](hidden_channels=hidden_channels, num_conv_layers=num_conv_layers, include_meta_node=include_meta_node)
        decoder = m.MLP([hidden_channels, hidden_channels, 1])
        
        # Load model
        model_path = f"./best_model_params/m_{test_model_param}.pt"
        decoder_path = f"./best_model_params/d_{test_model_param}.pt"
        model.load_state_dict(torch.load(model_path))
        decoder.load_state_dict(torch.load(decoder_path))
        model.to(device)
        decoder.to(device)

        # Test
        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for batched_data in test_loader:
                batched_data = batched_data.to(device)
                out = model(batched_data.x_dict, batched_data.deg_dict, batched_data.edge_index_dict)
                out = scatter_mean(out, batched_data["variable"]["batch"], dim=0)
                out = decoder(out).sigmoid()
                labels = np.concatenate([labels, batched_data.y.T.cpu().numpy()[0]], 0)
                predictions = np.concatenate([predictions, torch.round(out).T.cpu().numpy()[0]], 0)

        print("Testing Complete.")
        accuracy = np.sum(predictions == labels) / num_testing_instances
        precision = np.sum(predictions[labels == 1] == 1) / np.sum(predictions == 1)
        recall = np.sum(predictions[labels == 1] == 1) / np.sum(labels == 1)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")

        results["datetime"].append(pd.Timestamp.now().replace(microsecond=0))
        results["model_parameters"].append(test_model_param)
        results["problem_types"].append(test_problem_type)
        results["difficulty_levels"].append(test_difficulty_level)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)

    np.savetxt("test_predictions.txt", predictions)
    np.savetxt("test_labels.txt", labels) 
    results_pd = pd.DataFrame.from_dict(results)
    results_pd = results_pd.astype("str")
    results_pd.fillna("nan", inplace=True)
    results_pd.to_csv("./best_model_params/0test_results.csv", mode='a', header=True, index=False)



if __name__ == "__main__":
    main()

