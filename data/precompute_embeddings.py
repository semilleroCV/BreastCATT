# precompute_embeddings.py
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings with GatorTron-Base Language Model and upload them to Hugging Face with the rest of the data")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SemilleroCV/DMR-IR",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument("--split", type=str, default="train",
                        help="The split of the dataset we want to get text embeddings.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. Load your dataset and language model
    print("Loading dataset and model...")
    # Use the same model as in your tfvit.py
    lm_name = "UFNLP/gatortron-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    language_model = AutoModel.from_pretrained(lm_name).to(device)
    language_model.eval() # Set to evaluation mode

    # Load your dataset from the Hub
    dataset = load_dataset(args.dataset_name, split=args.split) # Or whichever split you need

    # 2. Define the function to compute embeddings
    def compute_embedding(batch):
        # Assuming your text is in a column named 'text'
        # Adjust 'text' to your actual column name if different
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = language_model(**inputs)
            # Using pooler_output, same as in your LanguageModel class
            embedding = outputs.pooler_output
        # The map function requires a list of arrays
        return {"text_embedding": embedding.cpu().numpy()}

    # 3. Map the function over the dataset
    # This will add a new column 'text_embedding'
    print("Computing embeddings...")
    # Use a larger batch size for efficiency
    dataset_with_embeddings = dataset.map(compute_embedding, batched=True, batch_size=32)

    # 4. Push the new dataset to the Hub
    print("Pushing new dataset to the Hub...")
    # Make sure you are logged in: `huggingface-cli login`
    # You can save it as a new dataset or a new configuration of the existing one.
    dataset_with_embeddings.push_to_hub(args.dataset_name, config_name="with_embeddings")
    
    print("Done!")

if __name__ == "__main__":
    main()