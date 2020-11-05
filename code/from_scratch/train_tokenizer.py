from tokenizers import BertWordPieceTokenizer


def train(data):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=data, vocab_size=52_000, min_frequency=2)
    return tokenizer


if __name__ == "__main__":
    import argparse
    import sys
    import os

    command_line = " ".join(sys.argv[1:])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--output_dir_name", type=str, help="Output directory")
    args = parser.parse_args()

    tokenizer = train(args.train_data)
    tok_dir = args.output_dir_name + "/tokenizers"
    if not os.path.exists(tok_dir):
        os.mkdir(tok_dir)
    tokenizer.save_model(tok_dir)
