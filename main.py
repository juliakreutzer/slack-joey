import argparse

from bot import run_bot


def main():
    ap = argparse.ArgumentParser("Joey NMT Bot")

    ap.add_argument("model_dir", type=str,
                    help="Model directory with checkpoint for predictions.")

    ap.add_argument("--bpe_src_code", type=str,
                    help="path for bpe codes (if source is not pre-processed).")

    ap.add_argument("--tokenize", action="store_true",
                    help="Tokenize inputs with Moses tokenizer.")

    args = ap.parse_args()

    run_bot(model_dir=args.model_dir, bpe_src_code=args.bpe_src_code,
            tokenize=args.tokenize)

if __name__ == "__main__":
    main()
