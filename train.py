import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import optim as optima
from sklearn.utils import shuffle
from model import CPL
from load_data.load_HGNN_data import generate_G_from_H
from load_data.load_DGCN_data import generate_adj_matrices
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.loss_function import LossFunc
from utils.eval_performance import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=768, help='Input channel size')
parser.add_argument('--out_channels1', type=int, default=256, help='Output channel size for first layer')
parser.add_argument('--out_channels2', type=int, default=256, help='Output channel size for second layer')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--kd_loss_weight', type=float, default=1E-5, help='Distillation loss weight')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1E-3, help='Weight decay')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--T', type=float, default=0.5, help='Temperature coefficients')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--dataset', type=str, default='MOOC', help='dataset name.')
args = parser.parse_args()

logger = set_logger(args.dataset)    # Set up logging


def train(args):
    set_seed(args.seed)    # Set random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('Dataset: ' + args.dataset + '\n')

    logger.info("Read data complete!")
    # Load training, validation, and test data
    train_data_df = pd.read_csv(f'./data/{args.dataset}/train.csv', header=0)
    val_data_df = pd.read_csv(f'./data/{args.dataset}/val.csv', header=0)
    test_data_df = pd.read_csv(f'./data/{args.dataset}/test.csv', header=0)
    # Convert data into tuples
    train_data = [tuple(x) for x in train_data_df.to_numpy()]
    val_data = [tuple(x) for x in val_data_df.to_numpy()]
    test_data = [tuple(x) for x in test_data_df.to_numpy()]

    # Load concept data and number of concepts
    concept_df = pd.read_csv(f'./data/{args.dataset}/concepts_index.csv', header=None)
    num_concepts = len(concept_df)

    # Load concept-resource hypergraph and generate adjacency matrix
    adj = generate_G_from_H(pd.read_csv(f'./data/{args.dataset}/Hypergraph_H.csv', header=None))
    G = adj.to(device)

    # Load and process learning behavior graph adjacency matrices
    adj_out, adj_in = generate_adj_matrices(f'./data/{args.dataset}/clickStreamLink_data_id.csv', num_concepts)
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)

    # Load pre-trained BERT embeddings and convert to tensor
    bert_embeddings_df = pd.read_csv(f'./data/{args.dataset}/bert_embeddings.csv')
    embeddings = np.stack(bert_embeddings_df['bert_embedding'].apply(lambda x: np.fromstring(x, sep=',')).values)
    feature_matrix = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Initialize the model
    model = CPL(args.in_channels, args.out_channels1, args.out_channels2, G, adj_out, adj_in, feature_matrix, dropout_rate=args.dropout_rate).cuda()
    optimizer = optima.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = LossFunc(device, T=args.T)

    logger.info("Training!!!!")
    best_val_auc = 0.0    # Initialize the best validation AUC

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f'epoch: {epoch + 1}, lr = {optimizer.param_groups[0]["lr"]}')

        # Shuffle training data for each epoch
        X_train = np.array(shuffle(train_data, random_state=args.seed))
        sum_total_loss = 0.0

        model.train()

        batch_idx = 0
        for i in range(X_train.shape[0] // args.batch_size):
            x = X_train[batch_idx * args.batch_size: batch_idx * args.batch_size + args.batch_size]
            batch_idx += 1
            c1, c2 = x[:, 0], x[:, 1]
            target = x[:, -1]
            target = torch.tensor(target).to(device)

            optimizer.zero_grad()

            # Forward pass: Compute predictions from the model
            logit_H, logit_N, logit_T = model(c1, c2)

            # Calculate loss and perform backpropagation
            loss, loss_kd,  prediction, ground_truth = loss_func(logit_H, logit_N, logit_T, target[:, None].float())
            total_loss = loss + args.kd_loss_weight * loss_kd
            sum_total_loss += total_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()
            
        # Calculate average loss for the epoch
        average_loss = (sum_total_loss / batch_idx).float()
        logger.info(f"Average train loss for epoch {epoch + 1}: {average_loss.item()}")

        # Evaluate on training data
        train_metrics = evaluate(model, train_data, args.batch_size, device)
        logger.info(f"Train metrics: ACC = {train_metrics['ACC']:.4f}, F1 = {train_metrics['F1']:.4f}, "
                    f"Precision = {train_metrics['Precision']:.4f}, Recall = {train_metrics['Recall']:.4f}, "
                    f"AUC = {train_metrics['AUC']:.4f}, AP = {train_metrics['AP']:.4f}")

        # Evaluate on validation data
        val_metrics = evaluate(model, val_data, args.batch_size, device)
        logger.info(f"Validation metrics: ACC = {val_metrics['ACC']:.4f}, F1 = {val_metrics['F1']:.4f}, "
                    f"Precision = {val_metrics['Precision']:.4f}, Recall = {val_metrics['Recall']:.4f}, "
                    f"AUC = {val_metrics['AUC']:.4f}, AP = {val_metrics['AP']:.4f}")

        # Save the model if AUC improves on the validation set
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            logger.info(f'Saved new best model at epoch {epoch + 1} with val_auc: {best_val_auc:.4f}!!!')
            model_name = f"./best_model/{args.dataset}/{args.dataset}-DGCPL_best_net.pth"
            torch.save({
                'model': model.state_dict()
            }, model_name)
            logger.info(f"Model parameters saved to {model_name}!")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f'Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds\n')


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    total_time = end_time - start_time    # Total training time
    logger.info(f'Train time: {total_time:.2f} seconds\n')

