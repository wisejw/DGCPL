import time
import argparse
import numpy as np
import pandas as pd
import torch
from model import CPL
from load_data.load_HGNN_data import generate_G_from_H
from load_data.load_DGCN_data import generate_adj_matrices
from utils.eval_performance import evaluate_test


parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=768, help='Input channel size')
parser.add_argument('--out_channels1', type=int, default=256, help='Output channel size for first layer')
parser.add_argument('--out_channels2', type=int, default=256, help='Output channel size for second layer')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
# parser.add_argument('--kd_loss_weight', type=float, default=1E-6, help='Distillation loss weight')  # University_Course
# parser.add_argument('--kd_loss_weight', type=float, default=1E-1, help='Distillation loss weight')  # LectureBank
parser.add_argument('--kd_loss_weight', type=float, default=1E-5, help='Distillation loss weight')  # MOOC
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# parser.add_argument('--weight_decay', type=float, default=1E-4, help='Weight decay')  # University_Course
# parser.add_argument('--weight_decay', type=float, default=1E-2, help='Weight decay')  # LectureBank
parser.add_argument('--weight_decay', type=float, default=1E-3, help='Weight decay')  # MOOC
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--T', type=float, default=0.5, help='Temperature coefficients')
# parser.add_argument('--seed', type=int, default=25, help='Random seed')     # University_Course, LectureBank
parser.add_argument('--seed', type=int, default=42, help='Random seed')     # MOOC
# parser.add_argument('--dataset', type=str, default='University_Course', help='dataset name.')
# parser.add_argument('--dataset', type=str, default='LectureBank', help='dataset name.')
parser.add_argument('--dataset', type=str, default='MOOC', help='dataset name.')
args = parser.parse_args()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Dataset: ' + args.dataset + '\n')

    print("Read test data complete!")
    # Load the test data for the selected dataset
    # test_data_df = pd.read_csv(f'./data/{args.dataset}/test.csv', header=0)  # University_Course
    # test_data_df = pd.read_csv(f'./data/{args.dataset}/test.csv', header=0)  # LectureBank
    test_data_df = pd.read_csv(f'./data/{args.dataset}/test.csv', header=0)  # MOOC
    test_data = [tuple(x) for x in test_data_df.to_numpy()]

    # Load the concept data to get the number of concepts
    concept_df = pd.read_csv(f'./data/{args.dataset}/concepts_index.csv', header=None)
    num_concepts = len(concept_df)

    # Load the concept-resource hypergraph data
    adj = generate_G_from_H(pd.read_csv(f'./data/{args.dataset}/Hypergraph_H.csv', header=None))
    G = adj.to(device)

    # Load the learning behavior graph adjacency matrices for in-degree and out-degree connections
    adj_out, adj_in = generate_adj_matrices(f'./data/{args.dataset}/clickStreamLink_data_id.csv', num_concepts)
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)

    # Load pre-trained BERT embeddings
    bert_embeddings_df = pd.read_csv(f'./data/{args.dataset}/bert_embeddings.csv')
    embeddings = np.stack(bert_embeddings_df['bert_embedding'].apply(lambda x: np.fromstring(x, sep=',')).values)
    feature_matrix = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Initialize the model with the appropriate hyperparameters
    model = CPL(args.in_channels, args.out_channels1, args.out_channels2, G, adj_out, adj_in, feature_matrix, dropout_rate=args.dropout_rate).cuda()

    # Load the pre-trained model weights from the saved checkpoint
    model_name = f"./best_model/{args.dataset}/{args.dataset}-DGCPL_best_net.pth"
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Perform testing on the model
    print("Testing!!!!")
    test_metrics = evaluate_test(model, test_data, args.batch_size, device, save_path=f"./data/{args.dataset}/predictions_with_concepts.csv")
    # Print the evaluation metrics (Accuracy, F1 score, Precision, Recall, AUC, AP)
    print(f"Test metrics: ACC = {test_metrics['ACC']:.4f}, F1 = {test_metrics['F1']:.4f}, "
                f"Precision = {test_metrics['Precision']:.4f}, Recall = {test_metrics['Recall']:.4f}, "
                f"AUC = {test_metrics['AUC']:.4f}, AP = {test_metrics['AP']:.4f}")


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    total_time = end_time - start_time    # Calculate the total test time
    print(f'Test time: {total_time:.2f} seconds\n')

