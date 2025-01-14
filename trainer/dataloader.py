from datasets import load_dataset
from torch.utils.data import DataLoader



def load_datasets(args):
    recompute_log = False
    try:
        dataset = load_dataset(args.task.query_dataset + '_' + args.task.cluster, split='train')
        dataset = dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward", "chosen_logprob",
                                                        "reject_token", "reject_mask", "reject_reward", "reject_logprob"])
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)

        validation_dataset = load_dataset(args.task.query_dataset + '_' + args.task.cluster, split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward", "chosen_logprob",
                                                                              "reject_token", "reject_mask", "reject_reward", "reject_logprob"])

        #Check if the dataset has elements else recompute logprobs
        print("LENGTH OF DATASET", len(dataset))
        if len(dataset) == 0:
           recompute_log = True
    except:
        dataset = load_dataset(args.task.query_dataset, split='train')
        dataset = dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward",
                                                        "reject_token", "reject_mask", "reject_reward"])
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.query_dataset, split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=["chosen_token", "chosen_mask", "chosen_reward",
                                                                              "reject_token", "reject_mask", "reject_reward"])
        recompute_log = True
    return dataset, validation_dataset, temp_dataloader, recompute_log
