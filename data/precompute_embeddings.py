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
    parser.add_argument(
        "--base_config_name",
        type=str,
        default="with_full_clinical_metadata",
        help="The name of the base dataset configuration to load from the Hub."
    )
    parser.add_argument(
        "--new_config_name",
        type=str,
        default="with_updated_embeddings",
        help="The name for the new dataset configuration on the Hub."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. Load your dataset and language model
    print(f"Loading dataset '{args.dataset_name}' with config '{args.base_config_name}'...")
    # Use the same model as in your tfvit.py
    lm_name = "UFNLP/gatortron-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    language_model = AutoModel.from_pretrained(lm_name).to(device)
    language_model.eval() # Set to evaluation mode

    # Load your dataset from the Hub. This will be a DatasetDict.
    dataset_dict = load_dataset(args.dataset_name, name=args.base_config_name)

    # 2. Define the function to compute embeddings
    def compute_embedding(batch):
        # Assuming your text is in a column named 'text'
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = language_model(**inputs)
            # Using pooler_output
            embedding = outputs.pooler_output
        # The map function requires a list of arrays
        return {"text_embedding": embedding.cpu().numpy()}

    # 3. Map the function over the entire dataset dict
    # This will add/overwrite the 'text_embedding' column in all splits
    print("Computing and overwriting embeddings for all splits...")
    # Use a larger batch size for efficiency
    dataset_with_embeddings = dataset_dict.map(compute_embedding, batched=True, batch_size=32)

    # 4. Push the new dataset to the Hub
    # Make sure you are logged in: `huggingface-cli login`
    print(f"Pushing new dataset to the Hub under config: {args.new_config_name}")
    dataset_with_embeddings.push_to_hub(args.dataset_name, config_name=args.new_config_name)
    
    print("Done!")

if __name__ == "__main__":
    main()