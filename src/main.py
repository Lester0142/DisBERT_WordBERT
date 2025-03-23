import os
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from util.config import Config
from util.data_loader import load_train_data, load_test_data
from util.data_processor import PROCESSORS, OUTPUT_TASKS
from util.logger import Logger, create_log_dir
from util.main_helper import set_seed, load_pretrained_model, load_trained_model, run_train, run_eval


def main():

    # =================== Load Config File ===================
    config = Config(main_conf_path="./src/config/")
    args = config
    # =================== End of Load Config File ===================

    # =================== Create Directory for Model Checkpoints ===================
    if not os.path.exists("saves"):
        os.mkdir("saves")
    # =================== End of Create Directory for Model Checkpoints ===================

    # =================== Initialize Logger ===================
    log_dir = create_log_dir(os.path.join("saves", args.bert_model))
    config.save(log_dir)
    args.log_dir = log_dir
    logger = Logger(log_dir)
    logger.info(args.__dict__)
    # =================== End of Initialize Logger ===================
    
    # =================== Initialize CUDA ===================
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))
    # =================== End of Initialize CUDA ===================

    # =================== Initialize SEED ===================
    set_seed(args.seed, args.n_gpu)
    # =================== End of Initialize SEED ===================

    # =================== Load Dataset and Processors ===================
    task_name = args.task_name.lower()
    processor = PROCESSORS[task_name]()
    output_task = OUTPUT_TASKS[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)
    # =================== End of Load Dataset and Processors ===================

    # =================== Load Model & Tokenizer ===================
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = load_pretrained_model(args)
    # =================== End of Load Model & Tokenizer ===================

    ########### Training ###########
    # VUA18 / VUA20
    if args.do_train and args.task_name == "vua":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_task
        )
        model, best_result = run_train(
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_task,
        )
    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi":
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            model = load_pretrained_model(args)
            train_dataloader = load_train_data(
                args, logger, processor, task_name, label_list, tokenizer, output_task, k
            )
            model, best_result = run_train(
                args,
                logger,
                model,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_task,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = {k: sum(d[k] for d in k_result) / len(k_result) for k in k_result[0]}
        logger.info(f"-----Average Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")

    # =================== Load Trained Model ===================
    if "saves" in args.bert_model:
        model = load_trained_model(args, model)
    # =================== End of Load Trained Model ===================

    ########### Inference ###########
    # VUA18 / VUA20
    if (args.do_eval or args.do_test) and task_name == "vua":
        # if test data is genre or POS tag data
        if ("genre" in args.data_dir) or ("pos" in args.data_dir):
            if "genre" in args.data_dir:
                targets = ["acad", "conv", "fict", "news"]
            elif "pos" in args.data_dir:
                targets = ["adj", "adv", "noun", "verb"]
            orig_data_dir = args.data_dir
            for idx, target in tqdm(enumerate(targets)):
                logger.info(f"====================== Evaluating {target} =====================")
                args.data_dir = os.path.join(orig_data_dir, target)
                all_guids, eval_dataloader = load_test_data(
                    args, logger, processor, task_name, label_list, tokenizer, output_task
                )
                run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
        else:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_task
            )
            run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

    # TroFi / MOH-X (K-fold)
    elif (args.do_eval or args.do_test) and args.task_name == "trofi":
        logger.info(f"***** Evaluating with {args.data_dir}")
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_task, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
            k_result.append(result)

        # Calculate average result
        avg_result = {k: sum(d[k] for d in k_result) / len(k_result) for k in k_result[0]}
        logger.info(f"-----Average Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")

    logger.info(f"Saved to {logger.log_dir}")


if __name__ == "__main__":
    main()