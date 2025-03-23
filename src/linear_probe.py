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
    
    for i in range(n):
        all_features = []
        all_labels = []
        
        # Iterate over the entire dataloader to extract features and labels
        for data in dataloader:
            prompt = data["prompt"]  # Get the prompt text
            # Tokenize the prompt (with truncation if needed) and move tensors to device
            label = data["label"][0].to(device)
            
            with torch.no_grad():
                try:
                  inputs = tokenizer(prompt, return_tensors="pt").to(device)
                  outputs = model(**inputs)
                  # Assume model returns a 'pooler_output' attribute containing the embeddings
                  features = outputs.pooler_output
                except:
                  inputs = tokenizer(prompt, return_tensors="pt", max_length=77, truncation=True).to(device)
                  features = model.get_text_features(**inputs)
            all_features.append(features)
            all_labels.append(label.unsqueeze(0))
        
        # Concatenate all features and labels and convert to numpy arrays
        all_features = torch.cat(all_features).cpu().float().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        # Split the data into training and testing sets while preserving label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )
        
        # Train the linear probe using Logistic Regression
        classifier = LogisticRegression(random_state=0, C=0.4, max_iter=1000, verbose=0)
        classifier.fit(X_train, y_train)
        
        # Predict and compute accuracy
        predictions = classifier.predict(X_test)
        accuracy = np.mean((y_test == predictions).astype(float)) * 100.
        accuracies.append(accuracy)
        print(f"Iteration {i+1}/{n} - Accuracy: {accuracy:.2f}%")
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Average Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    return avg_accuracy, std_accuracy