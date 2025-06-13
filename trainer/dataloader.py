from datasets import load_dataset
from torch.utils.data import DataLoader



def load_datasets(args):
    recompute_log = False
    print("Loading dataset....")

    if args.training_method == "GRPO":
        columns = ["token", "mask", "payoff"]
        extended_columns = columns + ["logprob"]
    else:
        columns = ["chosen_token", "chosen_mask", "chosen_reward", "reject_token", "reject_mask", "reject_reward"]
        extended_columns = columns + ["chosen_logprob", "reject_logprob"]

    
    try:
        dataset = load_dataset(args.dataset + '_' , split='train')
        dataset = dataset.with_format("torch", columns=extended_columns)
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)

        validation_dataset = load_dataset(args.dataset + '_', split='test')
        validation_dataset = validation_dataset.with_format("torch", columns=extended_columns)

        #Check if the dataset has elements else recompute logprobs
        print("LENGTH OF DATASET", len(dataset))
        if len(dataset) == 0:
           recompute_log = True
    except:
        dataset = load_dataset(args.dataset, split='train')
        dataset = dataset.with_format("torch", columns=columns)
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        try:
            validation_dataset = load_dataset(args.dataset, split='test')
            validation_dataset = validation_dataset.with_format("torch", columns=columns)
        except:
            #make a new dataset with the same elements as the training dataset
            validation_dataset = dataset.select(range(len(dataset)//20))

        recompute_log = True
    return dataset, validation_dataset, temp_dataloader, recompute_log
