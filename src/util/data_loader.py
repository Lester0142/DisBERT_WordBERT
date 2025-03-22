import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from util.data_helper import (
    convert_examples_to_two_features,
    convert_examples_to_three_features,
)


def load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_task, k=None):
    # get train examples
    if task_name != "trofi" and task_name != "vua":
        raise ("task_name should be 'vua' or 'trofi'!")
    train_examples = processor.get_train_examples(args.data_dir, k)

    # create model features
    if args.model_name == "DisBERT":
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_task, args
        )
    elif args.model_name in ["WordBERT", "DefBERT"]:
        train_features = convert_examples_to_three_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_task, args
        )
    else:
        raise ("model_name should be 'DisBERT', 'WordBERT' or 'DefBERT'!")

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
    all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in train_features], dtype=torch.long)

    # add additional features
    if args.model_name == "DisBERT":
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )

    elif args.model_name in ["WordBERT", "DefBERT"]:
        all_input_ids_3 = torch.tensor([f.input_ids_3 for f in train_features], dtype=torch.long)
        all_input_mask_3 = torch.tensor([f.input_mask_3 for f in train_features], dtype=torch.long)

        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
            all_input_ids_3,
            all_input_mask_3,
        )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader


def load_dev_data(args, logger, processor, task_name, label_list, tokenizer, output_task, k=None):
    # get train examples
    if task_name != "trofi" and task_name != "vua":
        raise ("task_name should be 'vua' or 'trofi'!")
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir, k)
    else:
        eval_examples = processor.get_dev_examples(args.data_dir)

    if args.model_name == "DisBERT":
        eval_features = convert_examples_to_two_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_task, args
        )
    elif args.model_name in ["WordBERT", "DefBERT"]:
        eval_features = convert_examples_to_three_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_task, args
        )
    else:
        raise ("model_name should be 'DisBERT', 'WordBERT' or 'DefBERT'!")

    logger.info("***** Running evaluation *****")
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_guids = [f.guid for f in eval_features]
    all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
    all_input_mask_2 = torch.tensor([f.input_mask_2 for f in eval_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
    
    if args.model_name == "DisBERT":
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    elif args.model_name in ["WordBERT", "DefBERT"]:
        all_input_ids_3 = torch.tensor([f.input_ids_3 for f in eval_features], dtype=torch.long)
        all_input_mask_3 = torch.tensor([f.input_mask_3 for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
            all_input_ids_3,
            all_input_mask_3,
        )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader


def load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_task, k=None):
    # get train examples
    if task_name != "trofi" and task_name != "vua":
        raise ("task_name should be 'vua' or 'trofi'!")
    eval_examples = processor.get_test_examples(args.data_dir, k)

    if args.model_name == "DisBERT":
        eval_features = convert_examples_to_two_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_task, args
        )
    elif args.model_name in ["WordBERT", "DefBERT"]:
        eval_features = convert_examples_to_three_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_task, args
        )
    else:
        raise ("model_name should be 'DisBERT', 'WordBERT' or 'DefBERT'!")

    logger.info("***** Running evaluation *****")
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_guids = [f.guid for f in eval_features]
    all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
    all_input_mask_2 = torch.tensor([f.input_mask_2 for f in eval_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
    
    if args.model_name == "DisBERT":
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    elif args.model_name in ["WordBERT", "DefBERT"]:
        all_input_ids_3 = torch.tensor([f.input_ids_3 for f in eval_features], dtype=torch.long)
        all_input_mask_3 = torch.tensor([f.input_mask_3 for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
            all_input_ids_3,
            all_input_mask_3,
        )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader