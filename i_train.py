import torch
from torch.nn import BCELoss
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.auto import tqdm, trange
from itertools import product
import numpy as np
import argparse

from my_utils import formula_utils as F
from my_utils import graph_utils as f2g
import models as m

def main():
    parser = argparse.ArgumentParser(description="Train model and save best parameters.")
    parser.add_argument("-d", "--difficulty", type=str, choices=["easy", "medium", "hard"], required=True, nargs='+', help="easy, medium, or hard, (combinations may also be provided)")
    parser.add_argument("-p", "--problem_type", type=str, required=True, nargs='+', help="3-sat, or sr")
    parser.add_argument("-r", "--representation", type=str, choices=["VCGm", "VCG", "LCGm", "LCG"], required=True, help="VCGm, VCG, LCGm, or LCG")
    parser.add_argument("-m", "--model", type=str, choices=["GCN"], required=True, help="GCN")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=2, help="batch size for training")
    parser.add_argument("-l", "--latent_dim", type=int, required=False, default=2, help="dimension of latent space")
    parser.add_argument("-c", "--num_conv_layers", type=int, required=False, default=4, help="number of convolutional layers")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=5, help="number of epochs to train")
    parser.add_argument("-s", "--seed", type=int, required=False, default=42, help="random seed for consistency")
    parser.add_argument("--different", type=str, required=False, default="", help="differentiate between different runs of the same model")
    
    args = parser.parse_args()
    
    if type(args.difficulty) == list:
        train_difficulty_levels = args.difficulty
        val_difficulty_levels = args.difficulty
    else:
        train_difficulty_levels = [args.difficulty]
        val_difficulty_levels = [args.difficulty]

    if type(args.problem_type) == list:
        train_problem_types = args.problem_type
        val_problem_types = args.problem_type
    else:
        train_problem_types = [args.problem_type]
        val_problem_types = [args.problem_type]

    train_splits = ["train"]
    val_splits = ["valid"]
    satisfiability = ["sat", "unsat"]

    paths_to_training_instances = F.get_paths_to_instances(train_difficulty_levels, train_problem_types, train_splits, satisfiability, "./dataset")
    paths_to_validation_instances = F.get_paths_to_instances(val_difficulty_levels, val_problem_types, val_splits, satisfiability, "./dataset")

    # Read in data
    training_formulas = F.read_instances_from_paths(paths_to_training_instances)
    validation_formulas = F.read_instances_from_paths(paths_to_validation_instances)

    # Process data
    transform_fn = {"VCGm": f2g.convert_instance_to_VCG_bi_with_meta_node,
                    "VCG": f2g.convert_instance_to_VCG_bi,
                    "LCGm": f2g.convert_instance_to_LCG_with_meta_node,
                    "LCG": f2g.convert_instance_to_LCG}
    training_dataset = [transform_fn[args.representation](formula) for formula in training_formulas]
    validation_dataset = [transform_fn[args.representation](formula) for formula in validation_formulas]

    # Create data loaders
    num_training_instances = len(training_dataset)
    num_validation_instances = len(validation_dataset)
    print(f"Number of training instances: {num_training_instances}")
    print(f"Number of validation instances: {num_validation_instances}")
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    hidden_channels = args.latent_dim
    if args.representation in ["VCGm", "VCG"]:
        model_choices = {"GCN": m.G4GCN_VCG}
    elif args.representation in ["LCGm", "LCG"]:
        model_choices = {"GCN": m.G4GCN_LCG}

    if args.representation in ["VCGm", "LCGm"]:
        include_meta_node = True
    elif args.representation in ["VCG", "LCG"]:
        include_meta_node = False

    model = model_choices[args.model](hidden_channels=hidden_channels, num_conv_layers=args.num_conv_layers, include_meta_node=include_meta_node)
    decoder = m.MLP([hidden_channels, hidden_channels, 1])
    model.to(device)
    decoder.to(device)
    model_params = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr=1e-4, weight_decay=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    # Train model
    train_loss_record = []
    cumulative_val_loss_record = []
    best_val_loss = np.inf
    pbar = trange(args.epochs, desc="Epoch")
    for epoch in pbar:
        model.train()
        for batched_data in train_loader:
            batched_data = batched_data.to(device)
            optimizer.zero_grad()
            out = model(batched_data.x_dict, batched_data.deg_dict , batched_data.edge_index_dict)
            out = scatter_mean(out, batched_data["variable"]["batch"], dim=0)
            out = decoder(out).sigmoid()
            train_loss = BCELoss()(out, batched_data.y)
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                pbar.set_description(f"loss: {train_loss.item():.2f}")
                train_loss_record.append(train_loss.item())

        model.eval()
        cumulative_loss = 0
        predictions = []
        labels = []
        with torch.no_grad():
            for batched_data in val_loader:
                batched_data = batched_data.to(device)
                out = model(batched_data.x_dict, batched_data.deg_dict , batched_data.edge_index_dict)
                out = scatter_mean(out, batched_data["variable"]["batch"], dim=0)
                out = decoder(out).sigmoid()
                val_loss = BCELoss()(out, batched_data.y)
                predictions = np.concatenate([predictions, torch.round(out).T.cpu().numpy()[0]], 0)
                labels = np.concatenate([labels, batched_data.y.T.cpu().numpy()[0]], 0)
                cumulative_loss += val_loss.item()
            cumulative_val_loss_record.append(cumulative_loss)
            if cumulative_loss < best_val_loss:
                best_predictions = predictions
                best_labels = labels
                best_val_loss = cumulative_loss
                torch.save(model.state_dict(), f"./best_model_params/m_{args.model}_{args.representation}_{args.problem_type}_{'and'.join(args.difficulty)}_ld{str(args.latent_dim)}_c{str(args.num_conv_layers)}_{args.different}.pt")
                torch.save(decoder.state_dict(), f"./best_model_params/d_{args.model}_{args.representation}_{args.problem_type}_{'and'.join(args.difficulty)}_ld{str(args.latent_dim)}_c{str(args.num_conv_layers)}_{args.different}.pt")
        scheduler.step(cumulative_loss)
    
    np.savetxt(f"train_loss_record",np.array(train_loss_record))
    np.savetxt("cumulative_val_loss_record",np.array(cumulative_val_loss_record))
    print("Training complete.")
    print(f"Training BCE Losses - min: {min(train_loss_record):.2f}, max: {max(train_loss_record):.2f}")
    accuracy = np.sum(best_predictions == best_labels) / num_validation_instances
    print(f"Best validation accuracy - {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
