from src.tokenization.bpe_tokenizer import BPETokenizer
import numpy as np
import os
# path of corpus to tokenize
config = {
    "tiny_stories": {
        "weight_path": "./src/tokenization/saved_bpe_tiny_story_train.json",
        "data_path": [
            "./data/TinyStoriesV2-GPT4-valid.txt",
            "./data/TinyStoriesV2-GPT4-train.txt"
        ]
    },
    "owt": {
        "weight_path": "./src/tokenization/saved_bpe_owt_train.json",
        "data_path": [
            "./data/owt_train.txt",
            "./data/owt_valid.txt"
        ],
    }
}

def main():
    for name in config:
        print(f"name: {name}")
        # load weights
        bpe_tokenizer = BPETokenizer.load(config[name]["weight_path"])

        # load corpus and encode
        for data_path in config[name]["data_path"]:
            corpus = ""
            with open(data_path, 'r') as f:
                corpus = f.read()
            token_list = bpe_tokenizer.encode(corpus, use_parallel=True, num_processes=4)

            # save token_list as numpy array of type uint16
            token_list = np.array(token_list, dtype=np.uint16)
            np.save(data_path.replace('.txt', '_tokens.npy'), token_list)

if __name__ == '__main__':
    main()    
