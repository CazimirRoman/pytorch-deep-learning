def show_image_from_loader(loader, batch_idx=0, idx=0, mean=None, std=None):
    """
    Function to display an image from a DataLoader batch.
    
    Args:
    - loader (DataLoader): PyTorch DataLoader object containing the dataset.
    - batch_idx (int): Index of the batch to visualize (default: 0).
    - idx (int): Index of the image within the batch to visualize (default: 0).
    - mean (list): Mean values used for normalization (default: None).
    - std (list): Standard deviation values used for normalization (default: None).
    
    Note:
    - If mean and std are provided, the function will perform unnormalization.
    """
    # Get the specified batch from DataLoader
    batch_idx, (X_batch, y_batch) = next(enumerate(loader))
    
    # Extract the specified image tensor from X_batch
    image = X_batch[idx]

    # If mean and std are provided, perform unnormalization
    if mean is not None and std is not None:
        unnormalize = transforms.Normalize(mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
                                           std=[1 / std[0], 1 / std[1], 1 / std[2]])
        image = unnormalize(image)

    # Convert tensor to numpy array
    image_np = image.cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # Reorder dimensions for matplotlib

    # Display the image
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()