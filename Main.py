import os
import random
import yaml
from ultralytics import YOLO
from shutil import copyfile

def split_dataset(images_path, labels_path, output_dir, train_ratio=0.8):
    """
    Splits a dataset of images and labels into training and validation sets.

    Args:
        images_path (str): Path to the directory containing the image files.
        labels_path (str): Path to the directory containing the label files (in YOLO format).
        output_dir (str): Path to the directory where the split dataset will be saved.
        train_ratio (float, optional): The ratio of data to use for training (default: 0.8).
    """
    # Create output directories if they don't exist
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Check if the image and label paths exist
    if not os.path.exists(images_path):
        print(f"Error: Image path '{images_path}' not found.")
        return None, None, None, None  # Return None values to indicate failure
    if not os.path.exists(labels_path):
        print(f"Error: Label path '{labels_path}' not found.")
        return None, None, None, None

    # Get list of image files (assuming common image formats)
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    label_files = [os.path.splitext(f)[0] + '.txt' for f in image_files] # Assumes labels have same name as image but with .txt

    # Check if label file exists for each image file.
    for label_file in label_files:
        if not os.path.exists(os.path.join(labels_path, label_file)):
            print(f"Error: Label file {label_file} not found in {labels_path}.  Skipping image {os.path.splitext(label_file)[0]}.")
            if os.path.splitext(label_file)[0] + os.path.splitext(image_files[label_files.index(label_file)])[1] in image_files:
                image_files.remove(os.path.splitext(label_file)[0] + os.path.splitext(image_files[label_files.index(label_file)])[1])
            label_files.remove(label_file)
            
    if not image_files:
        print("Error: No image files found after checking for corresponding label files.")
        return None, None, None, None

    # Shuffle the files to ensure random distribution
    combined_files = list(zip(image_files, label_files))
    random.shuffle(combined_files)
    image_files, label_files = zip(*combined_files) # Unzip

    # Calculate the split index
    split_index = int(len(image_files) * train_ratio)

    # Copy files to their respective directories
    for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        src_image_path = os.path.join(images_path, image_file)
        src_label_path = os.path.join(labels_path, label_file)

        if i < split_index:  # Training set
            dst_image_path = os.path.join(train_images_dir, image_file)
            dst_label_path = os.path.join(train_labels_dir, label_file)
        else:  # Validation set
            dst_image_path = os.path.join(val_images_dir, image_file)
            dst_label_path = os.path.join(val_labels_dir, label_file)

        copyfile(src_image_path, dst_image_path)
        copyfile(src_label_path, dst_label_path)

    print(f"Dataset split complete.  Training images: {len(image_files[:split_index])}, Validation images: {len(image_files[split_index:])}")
    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir



def create_updated_yaml(original_yaml_path, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, output_yaml_path):
    """
    Creates a new data.yaml file with updated paths to the training and validation sets.

    Args:
        original_yaml_path (str): Path to the original data.yaml file.
        train_images_dir (str): Path to the directory containing training images.
        train_labels_dir (str): Path to the directory containing training labels.
        val_images_dir (str): Path to the directory containing validation images.
        val_labels_dir (str): Path to the directory containing validation labels.
        output_yaml_path (str): Path to where the new data.yaml file should be saved.
    """
    # Load the original YAML file
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Update the paths.  Ultralytics wants *directories*, not files.
    data['train'] = os.path.dirname(train_images_dir)  # The parent directory of the images
    data['val'] = os.path.dirname(val_images_dir)    # The parent directory of the images

    # Write the updated YAML data to a new file
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f)
    print(f"Updated data.yaml file created at: {output_yaml_path}")
    return output_yaml_path



if __name__ == '__main__':
    # --- Configuration ---
    original_dataset_yaml_path = 'D:/Dataset/Pistols.v1-resize-416x416.yolov11/data.yaml'  # Replace with your original dataset YAML path
    images_path = 'D:/Dataset/Pistols.v1-resize-416x416.yolov11/dataset/images' # Path to the directory containing ALL your images
    labels_path = 'D:/Dataset/Pistols.v1-resize-416x416.yolov11/dataset/labels' # Path to the directory containing ALL your labels
    output_dir = 'D:/Dataset/Pistols.v1-split'  # Directory to save the split dataset
    updated_yaml_path = 'D:/Dataset/Pistols.v1-split/data_split.yaml' # Where to save the new yaml file
    pretrained_model = 'yolo11m.pt'
    epochs = 100
    image_size = 640
    batch_size = 16
    learning_rate = 0.01
    optimizer = 'AdamW'
    project_name = 'yolov11_custom_training'
    experiment_name = 'run1'

    # --- Split the dataset ---
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = split_dataset(images_path, labels_path, output_dir)
    if train_images_dir is None:
        print("Dataset splitting failed.  Exiting.")
        exit()

    # --- Create updated YAML file ---
    updated_yaml_path = create_updated_yaml(original_dataset_yaml_path, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, updated_yaml_path)


    # --- Load the YOLO model ---
    try:
        model = YOLO(pretrained_model)
    except Exception as e:
        print(f"Error loading pretrained model '{pretrained_model}': {e}")
        print("Make sure the Ultralytics library is up to date and the model name is correct.")
        exit()

    # --- Train the model ---
    try:
        results = model.train(
            data=updated_yaml_path, # Use the path to the updated YAML file
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            lr0=learning_rate,
            optimizer=optimizer,
            project=project_name,
            name=experiment_name
        )

        # --- Training Results ---
        print("\nTraining complete!")
        print(f"Results saved to: {results.save_dir}")

    except Exception as e:
        print(f"Error during training: {e}")
