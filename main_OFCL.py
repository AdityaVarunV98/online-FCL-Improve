import torch
import numpy as np

from configuration import config_FCL
from utils.data_loader import get_loader_all_clients
from utils.train_utils import get_free_gpu_idx, get_logger, initialize_clients, FedAvg, weightedFedAvg, test_global_model, save_results, FedAvg_split
from datetime import datetime

args = config_FCL.base_parser()
logger = get_logger(args)

if torch.cuda.is_available():
    gpu_idx = get_free_gpu_idx()
    args.cuda = True
    args.device = f'cuda:{gpu_idx}'
else:
    args.device = 'cpu' 

print(args)
print(args.device)

for run in range(args.n_runs):
    # Function that considers the directories, assigns data for the tasks and the clients
    # May not require any changes here
    loader_clients, cls_assignment_list, global_test_loader = get_loader_all_clients(args, run)
    
    # Defining the clients using the client task (make changes in the client)
    clients = initialize_clients(args, loader_clients, cls_assignment_list, run)

    start_time = datetime.now()
    # c=0
    comm_round = 0
    
    # print(clients[0].model)

    while not all([client.train_completed for client in clients]):
        for client in clients:
            if not client.train_completed:
                samples, labels = client.get_next_batch()
                if samples is not None:
                    if args.with_memory:
                        if client.task_id == 0:
                            # Require changes here to train with the new loss (change the training_step function)
                            client.train_with_update(samples, labels)
                        else:
                            # Require changes here to train with the new loss (change the training_step function)
                            client.train_with_memory(samples, labels)
                    else:
                        # if not c%100:
                        #     client.train(samples, labels, print_loss = True)
                        # else:
                        client.train(samples, labels, print_loss = False)
                        # c+=1
                else:
                    # print(f'Run {run} - Client {client.client_id} - Task {client.task_id} completed - {client.get_current_task()}')
                    class_counts = client.get_task_class_counts(client.task_id)
                    print(f'Run {run} - Client {client.client_id} - Task {client.task_id} completed - {client.get_current_task()}')
                    print(f'Class counts: {class_counts}')

                    
                    # compute loss train (just adds and shows the loss, no changes here)
                    logger = client.compute_loss(logger, run)
                    print(f'Run {run} - Client {client.client_id} - Test time - Task {client.task_id}')
                    
                    # Make sure the inference is done as per the paper here
                    logger = client.test(logger, run)
                    logger = client.validation(logger, run)
                    logger = client.forgetting(logger, run)

                    if client.task_id + 1 >= args.n_tasks:
                        client.train_completed = True
                        print(f'Run {run} - Client {client.client_id} - Train completed')
                        logger = client.balanced_accuracy(logger, run)
                    else:
                        client.task_id += 1

        # print("Number of comm rounds so far:", comm_round)
        # for client in clients:
        #     print(f"client {client.client_id}, batch number {client.batches_total}")

        for client in clients:
            if client.train_completed == False:
                total_batches = client.batches_total
                break

        # COMMUNICATION ROUND PART
        if args.train_completed_fed == "base":
            selected_clients = [client.client_id for client in clients if (client.num_batches >= args.burnin and client.batches_total % args.jump == 0 and client.train_completed == False)]
        else:
            selected_clients = [client.client_id for client in clients if (client.num_batches >= args.burnin and client.batches_total % args.jump == 0) or (client.train_completed == True and total_batches % args.jump == 0)]
        
        # Debugging
        # if len(selected_clients) > 1:
        #     print(total_batches)
        #     print("Selected Clients")
        #     for client_id in selected_clients:
        #         print(f"client {client_id}: train completed? {clients[client_id].train_completed}")
        #     print(f"Number of communication rounds: {comm_round}")

        if len(selected_clients) > 1:
            comm_round += 1  # keep a counter in args

            # --- PRE-AGGREGATION eval ---
            if comm_round % args.eval_gap == 0 or comm_round % args.eval_gap == 1:
                for cid in selected_clients:
                    metrics = clients[cid].intermediate_test(run, stage="pr", comm_round=comm_round)
                    print(f"Pre-agg round {comm_round}, client {cid}, accs: {metrics}")

            if args.fl_update == "split":
                split_strategy = {
                    "body": "fedavg",
                    "classifier": "weighted"
                }
                global_model = FedAvg_split(args, selected_clients, clients, split_strategy)
            elif args.fl_update.startswith("w_"):
                global_model = weightedFedAvg(args, selected_clients, clients)
            else:
                global_model = FedAvg(args, selected_clients, clients)

            global_parameters = global_model.state_dict()

            # update local models (redundant loops, but works)
            if args.train_completed_fed == "no_update":
                for client_id in selected_clients:
                    if clients[client_id].train_completed == False:
                        clients[client_id].save_last_local_model()
                        clients[client_id].update_parameters(global_parameters)
                        clients[client_id].save_last_global_model(global_model)
            else:
                for client_id in selected_clients:
                    clients[client_id].save_last_local_model()
                    clients[client_id].update_parameters(global_parameters)
                    clients[client_id].save_last_global_model(global_model)

            

            # --- POST-AGGREGATION eval ---
            if comm_round % args.eval_gap == 0:
                for cid in selected_clients:
                    metrics = clients[cid].intermediate_test(run, stage="po", comm_round=comm_round)
                    print(f"Post-agg round {comm_round}, client {cid}, accs: {metrics}")

    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}')
    print(f"Total communication rounds: {comm_round}")
    # global model accuracy when all clients finish their training on all tasks (FedCIL ICLR2023)
    logger = test_global_model(args, global_test_loader, global_model, logger, run)

final_accs = []
final_fors = []
for client_id in range(args.n_clients):
    print(f'Client {client_id}: {clients[client_id].task_list}')
    print(np.mean(logger['test']['acc'][client_id], 0))
    final_acc = np.mean(np.mean(logger["test"]["acc"][client_id], 0)[args.n_tasks-1,:], 0)
    final_for = np.mean(logger["test"]["forget"][client_id])
    final_accs.append(final_acc)
    final_fors.append(final_for)
    print(f'Final client accuracy: {final_acc}')
    print(f'Final client forgetting: {final_for}')
    print(f'Final client balanced accuracy: {np.mean(logger["test"]["bal_acc"][client_id])}')
    print()

print(f'Final average accuracy: {np.mean(final_accs):0.4f} (+-) {np.std(final_accs):0.4f}')
print(f'Final average forgetting: {np.mean(final_fors):0.4f} (+-) {np.std(final_fors):0.4f}')
print()

# save training results
save_results(args, logger)