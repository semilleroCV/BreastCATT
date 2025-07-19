def evaluate_linear_probe(n, dataloader, model, tokenizer, device):
    """
    Performs model forward passes n times, extracts embeddings, trains a Logistic Regression linear probe,
    and computes the average accuracy along with its standard deviation.

    Parameters:
      n           (int): Number of iterations to run the evaluation.
      dataloader  (iterable): DataLoader yielding dictionaries with keys "prompt" and "label".
      model       (torch.nn.Module): Model with a forward method returning 'pooler_output'.
      tokenizer   (transformers.PreTrainedTokenizer): Tokenizer to process the text inputs.
      device      (torch.device): Device (CPU or GPU) to perform computation on.
      
    Returns:
      avg_accuracy (float): Average accuracy across iterations (%).
      std_accuracy (float): Standard deviation of the accuracies.
    """
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    accuracies = []
    all_features = []
    all_labels = []

    # --- Feature Extraction ---
    # This part should be done only once.
    print("Extracting features from the model...")
    for data in dataloader:
        prompts = data["text"]  # Expect a batch of prompts
        labels = data["label"].to(device)
        
        with torch.no_grad():
            
            # Check if the model is a CLIP-like model which has a `get_text_features` method.
            if hasattr(model, 'get_text_features'):
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                # Use get_text_features for CLIP-like models
                features = model.get_text_features(**inputs)
            else:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                # For standard text encoders (BERT, GatorTron, etc.)
                outputs = model(**inputs)
                # Use pooler_output for classification tasks
                features = outputs.pooler_output

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate all features and labels from all batches
    all_features = torch.cat(all_features).float().numpy()
    all_labels = torch.cat(all_labels).numpy()

    # --- N-fold Evaluation ---
    print(f"Performing {n}-iteration evaluation...")
    for i in range(n):
        # Split the data into training and testing sets
        # A different random_state is used in each iteration for variability
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=i, stratify=all_labels
        )
        
        # Train the linear probe
        classifier = LogisticRegression(random_state=42, C=0.4, max_iter=1000, solver='liblinear')
        classifier.fit(X_train, y_train)
        
        # Evaluate and store accuracy
        accuracy = classifier.score(X_test, y_test) * 100.
        accuracies.append(accuracy)
        print(f"Iteration {i+1}/{n} - Accuracy: {accuracy:.2f}%")
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"\nAverage Accuracy over {n} runs: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    return avg_accuracy, std_accuracy