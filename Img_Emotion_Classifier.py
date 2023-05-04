import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix, cohen_kappa_score

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 200
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data PreProcessing Function:
def data_preprocessing():
    train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), #Pre-trained model expects rgb
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.Resize((224, 224)), # want a small standard image size to reduce load on GPU
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize(mean_pixel_val, std_pixel_val)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # PyTorch Official Documentation
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), #Pre-trained model expects rgb,
        transforms.Resize((224, 224)), # want a small standard image size to reduce load on GPU
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize(mean_pixel_val, std_pixel_val)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # PyTorch Official Documentation
    ])
    return train_transform, test_transform

def build_dataset(train_path, test_path, train_transform, test_transform):
    train_dataset = ImageFolder(train_path, train_transform)
    test_data = ImageFolder(test_path, test_transform)
    # Half the test data is split between testing and validation set.
    val_size = len(test_data) // 2
    test_size = len(test_data) - val_size
    # Randomly choose what data goes into test_set and validation_set
    val_dataset, test_dataset = random_split(test_data, [val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    # Print the class-to-index mappings
    print(train_dataset.class_to_idx)
    label_map = train_dataset.class_to_idx
    return train_dataset, val_dataset, test_dataset, label_map

def get_batch(batch):
    # Convert batch of tuples to a tuple of tensors
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return inputs, labels

def visualize_data(loader, num=2):
    # Get a batch of images and labels from the train_loader
    images, labels = next(iter(loader))

    # Get the first image and its label from the batch
    for i in range(num):
        image, label = images[i], labels[i]

        # Convert the image tensor to a numpy array and transpose it to the correct shape
        image = np.transpose(image.numpy(), (1, 2, 0))

        # Scale the pixel values to the range [0, 1]
        image = (image * 0.5) + 0.5

        # Plot the image and its label
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()
    plt.show()

def get_model():
    # VGG #
    # model = models.vgg19(weights='VGG19_Weights.DEFAULT') 
    # num_features = model.classifier[-1].in_features # num of features in last fully connected model layer
    # model.classifier[-1] = model.fc = nn.Sequential(
    #    nn.Dropout(0.4), #0.25, 0.4 -->67% ,0.5 --> val_acc lower than with no dropout
    #    nn.Linear(num_features, 7)
    # ) # 7 unique emotions in both train and test folders resp

    # ResNeXT # https://pytorch.org/vision/stable/models/resnext.html
    # model = models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.DEFAULT')
    # num_features = model.fc.in_features
    # # model.fc = nn.Linear(num_features, 7)
    # model.fc = nn.Sequential(
    #    nn.Dropout(0.6), 
    #    nn.Linear(num_features, 7)
    # )

    # ResNeXT # https://pytorch.org/vision/stable/models/resnext.html
    model = models.resnext101_64x4d(weights='ResNeXt101_64X4D_Weights.DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), #0.25, 0.4 -->67% ,0.5 --> val_acc lower than with no dropout
        nn.Linear(num_features, 7)
        )
    return model

def hyperparams(model):
    loss = nn.CrossEntropyLoss()

    
    optimizer = torch.optim.SGD(model.parameters(), 
                        lr=1e-3,   
                        momentum=0.99, 
                        nesterov=True, 
                        weight_decay=5e-4)
    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 min_lr=1e-8,
                                                 mode='min',
                                                 factor=0.45, 
                                                 patience=5, 
                                                 verbose=True) # reduce learning rate when loss plateaus
    
    return loss, optimizer, scheduler

def train_model(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, EPOCHS):
    # Define variables to keep track of the best validation accuracy and the corresponding model state
    best_acc = 0.0
    best_state_dict = None

    # Define early stopping variables
    patience = 10
    count = 0

    # Initialize GradScaler
    scaler = GradScaler()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []


    for epoch in range(EPOCHS):
        epoch_train_losses = []
        epoch_train_accs = []

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('--' * 20)

        model.train()  # Set the model to training mode

        for i, (inputs, labels) in enumerate(train_loader, 0): #enumerate(object, obj_idx)
            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Apply mixed-precision training
            with autocast():
                # Forward + Backward + Optimize
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update training statistics
            _, predicted = torch.max(outputs.data, 1)
            train_total = labels.size(0)
            train_correct = (predicted == labels).sum().item()
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(100 * train_correct / train_total)
            

            # Print statistics --- after every 100 batches
            if i % 100 == 99:
                print('Batch Num: [%5d] Batch_train_loss: %.3f, Batch_train_acc: %d %%' %
                    (i + 1, sum(epoch_train_losses[-100:]) / 100, sum(epoch_train_accs[-100:]) / 100))

        # Evaluate the model on the validation dataset
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        valid_acc = 100 * val_correct / val_total
        valid_loss = val_loss / len(val_loader)
        # Print validation loss and accuracy
        print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(valid_loss, valid_acc))

        train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
        train_accs.append(sum(epoch_train_accs) / len(epoch_train_accs))
        val_losses.append(valid_loss)
        val_accs.append(valid_acc)

        # Update the learning rate scheduler
        scheduler.step(valid_loss) 

        # Save the model checkpoint
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_acc": valid_acc,
        }
        
        checkpoint_path = f"/home/vedant02/SchoolWork/DATA/Chkpt_File/checkpoint-{epoch+1}_.bin"
        torch.save(checkpoint_dict, checkpoint_path)

        # Update the best model state and save the best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, "/home/vedant02/SchoolWork/DATA/Best_Model/resnext101_64X4D_drop_05_SGD_2.pth")

            # Reset the count for early stopping
            count = 0
        else:
            count += 1
            if count == patience:
                print(f"No improvement in test accuracy for {patience} epochs. Early stopping...")
                break

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Epoch vs Loss')
    plt.savefig("/home/vedant02/SchoolWork/resnext_101_drop_045/EpochvsLoss_ResNext101_5.png") #1 --> 045


    # Plot training and validation accuracy
    plt.figure()
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Epoch vs Accuracy')
    plt.savefig("/home/vedant02/SchoolWork/resnext_101_drop_045/EpochvsAccuracy_ResNext101_5.png") 
    print('... Training Completed ...')

def test_model(model_path, test_loader, device):
    # Load the best model
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluate the best model on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # Move data to the device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy on the Test Dataset: %d %%' % (100 * correct / total))

def predict_label(image_path, model, device, label_map):
    # Open the image
    image = Image.open(image_path)
    
    _,test_tranform = data_preprocessing()
    # Preprocess the image
    image = test_tranform(image)

    # Add a batch dimension to the image
    image = image.unsqueeze(0)

    # Move the image to the device
    image = image.to(device)
    model.to(device)

    # Disable autograd
    with torch.no_grad():
        # Predict the label
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Return the predicted label
    idx_to_class = {v: k for k, v in label_map.items()}
    predicted_label = idx_to_class[predicted.item()]
    return predicted_label

def show_confusion_matrix(cm):
  hmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(),  rotation = 0, ha = 'right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(),  rotation = 45, ha = 'right')
  plt.xlabel("PREDICTED COLUMNS")
  plt.ylabel("ACTUAL ROWS")
  plt.tight_layout() # center the plot within the figure
  plt.savefig("/home/vedant02/SchoolWork/resnext_101_drop_045/cm_resnext101_5.png")

def generate_classification_report(model, data_loader, device, label_map):
    # Move the model to the same device as the inputs
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    idx_to_class = {v: k for k, v in label_map.items()}
    

    # Initialize variables to store the ground truth labels and predicted labels
    y_true = []
    y_pred = []

    # Iterate over the data loader
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass to get the predicted outputs
            outputs = model(inputs)

            # Convert the outputs to predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Map the predicted labels to emotion labels using the label map
            predicted_labels = [idx_to_class[p.item()] for p in predicted]

            # Map the ground truth labels to emotion labels using the label map
            true_labels = [idx_to_class[label.item()] for label in labels]

            # Append the ground truth and predicted labels to the lists
            y_true += true_labels
            y_pred += predicted_labels

    # Generate the classification report
    target_names = list(label_map.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Compute the balanced accuracy score and confusion matrix
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)

    # Print the results
    print(f"Balanced accuracy score: {balanced_acc}")

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's kappa: {kappa}")
    show_confusion_matrix(df_cm)
    # print(f"Confusion matrix:\n{cm}")



def main():
    train_path = "/home/vedant02/SchoolWork/DATA/train"
    test_path = "/home/vedant02/SchoolWork/DATA/test"

    # Transformation Function to crop and resize all images along with other modifications
    # so that the model can learn diverse features.
    train_transform, test_transform = data_preprocessing()
    
    # Build a dataset from the train and test folders.
    train_dataset, val_dataset, test_dataset, label_map = build_dataset(train_path=train_path, test_path=test_path,
                                                             train_transform=train_transform, test_transform=test_transform)
    
    # Build data loader functions for efficient processing during training.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=get_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=get_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=get_batch)

    # Get the basic model
    model = get_model()
    model.to(device)

    # Advanced Hyperparams
    loss, optimizer, scheduler = hyperparams(model=model)

    # Train from checkpoint
    # chkpt = torch.load('/home/vedant02/SchoolWork/DATA/Chkpt_File/checkpoint-9_.bin')

    # print(chkpt.keys())
    # model.load_state_dict(chkpt['model_state_dict'])
    # optimizer.load_state_dict(chkpt['optimizer_state_dict'])

    # Training the Model
    train_model(model=model, optimizer=optimizer,
                scheduler=scheduler, loss_fn=loss,
                train_loader=train_loader, val_loader=val_loader,
                device=device, EPOCHS=EPOCHS)
    
    # Test the Model
    test_model(model_path="/home/vedant02/SchoolWork/DATA/Best_Model/resnext101_64X4D_drop_05_SGD_2.pth", test_loader=test_loader, device=device)
    
    # Predict single image
    # # image_path = '/home/vedant02/SchoolWork/DATA/test/happy/PrivateTest_95094.jpg'
    # # label = predict_label(image_path=image_path, model=model, device=device, label_map=label_map)
    # # # print(label)

    # Get evaluation_metrics
    # model =  get_model()
    # model.load_state_dict(torch.load('/home/vedant02/SchoolWork/DATA/Best_Model/resnext101_64X4D_drop_05_SGD.pth'))
    # generate_classification_report(model=model, data_loader=test_loader, device=device, label_map=label_map)
    
    

if __name__=="__main__":
  main()