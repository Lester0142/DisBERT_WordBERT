import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn 

from tqdm import tqdm, trange
from transformers import  AutoModel, AdamW, get_linear_schedule_with_warmup
from model.disbert import AutoModelForSequenceClassification_WSD_MIP_SPV
from model.wordbert import AutoModelForSequenceClassification_MIP_SPV_MIP2
from model.defbert import AutoModelForSequenceClassification_DEF
from util.data_loader import  load_test_data, load_dev_data
from util.metric import compute_metrics


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"

def set_seed(seed, n_gpu):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_pretrained_model(args):
    # Pretrained Model
    bert = AutoModel.from_pretrained(args.bert_model)
    config = bert.config
    config.type_vocab_size = 4
    embedding_size = config.embedding_size if "albert" in args.bert_model else config.hidden_size
    bert.embeddings.token_type_embeddings = nn.Embedding(config.type_vocab_size, embedding_size)
    bert._init_weights(bert.embeddings.token_type_embeddings)

    # Additional Layers
    if args.model_name == "DisBERT":
        model = AutoModelForSequenceClassification_WSD_MIP_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_name == "WordBERT":
        model = AutoModelForSequenceClassification_MIP_SPV_MIP2(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_name == "DefBERT":
        model = AutoModelForSequenceClassification_DEF(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model


def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)


def load_trained_model(args, model):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model

def run_train(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_task,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_scheduler != False or args.lr_scheduler.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(args.train_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_name in ["DisBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                ) = batch
            elif args.model_name in ["WordBERT", "DefBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                    input_ids_3,
                    input_mask_3,
                ) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            # compute loss values
            if args.model_name in ["DisBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            elif args.model_name in ["WordBERT", "DefBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    input_ids_3=input_ids_3,
                    attention_mask_3=input_mask_3,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_scheduler != False or args.lr_scheduler.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_dev_data(
                args, logger, processor, task_name, label_list, tokenizer, output_task, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result


def run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_name in ["DisBERT"]:
            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                idx,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
            ) = eval_batch
        elif args.model_name in ["WordBERT", "DefBERT"]:
            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                idx,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
                input_ids_3,
                input_mask_3,
            ) = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_name in ["DisBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )
            elif args.model_name in ["WordBERT", "DefBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    input_ids_3=input_ids_3,
                    attention_mask_3=input_mask_3,

                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result