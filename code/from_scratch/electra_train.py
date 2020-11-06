
#code from : https://towardsdatascience.com/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.language_modeling import LanguageModelingModel
import logging



if __name__ == "__main__":
    import argparse
    import sys
    import os
    command_line = " ".join(sys.argv[1:])


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--accu_step", default=64, type=int, help="Gradient acccumulation number of steps")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=2e-4, type=int, help="Learning rate")
    parser.add_argument("--adam_lr", default=1e-6, type=int, help="Adam epsilon")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="Number of warmup steps.")
    parser.add_argument("--eval_steps", default=76000, type=int, help="Evaluation step after n steps")
    parser.add_argument("--gpu", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--tokenizer",default=None, type=str, help="Path to tokenizer")
    parser.add_argument("--tok_data", default=None, type=str, help="Path to training data for tokenizer")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


    train_args = {
        # ? "fp16": True,
        "max_seq_length": 512,
        "train_batch_size": args.batch_size,
        # ?"eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accu_step,
        "num_train_epochs": args.epochs,
        # "weight_decay": 0.01,
        "learning_rate": args.lr,
        "adam_epsilon": args.adam_lr,
        # ? "warmup_ratio": 0.06,
        "warmup_steps": args.warmup_steps,
        # ? muzu to mit v upper case?
        "do_lower_case": True,
        "output_dir": "./results",
        "evaluate_during_training": True,
        "evaluate_during_training_steps": args.eval_steps,
        "save_eval_checkpoints": True,

        "reprocess_input_data": False,

        # ? "process_count": cpu_count() - 2 if cpu_count() > 2 else 1
        "n_gpu": args.gpu,
        "use_multiprocessing": True,

        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "early_stopping_delta": 0,
        "early_stopping_metric": "eval_loss",
        "early_stopping_metric_minimize": True,
        "overwrite_output_dir": True,

        "manual_seed": None,
        "encoding": None,
        "dataset_type": "simple",
        "tokenizer_name": args.tokenizer,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": True,
        "sliding_window": True,
        "vocab_size": 52000,
        "config": {
            "vocab_size": 52000,
        },
        "generator_config": {
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 3,
        },
        "discriminator_config": {
            "embedding_size": 768,
            "hidden_size": 768,
            "num_attention_heads": 12,
        },
    }

    train_file = "data/train_all.txt"
    test_file = "data/test.txt"

    model = LanguageModelingModel(
        "electra",
        None,
        args=train_args,
        use_cuda=False,
        train_files=args.tok_data
    )
    print(str(model.tokenizer))

    model.train_model(
        args.train_data, eval_file=args.test_data,
    )

    model.eval_model(test_file)



        