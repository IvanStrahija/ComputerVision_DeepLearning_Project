import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torchsummary
import os

checkpoint_dir = 'checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64
    

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    vgg19_bn = models.vgg19_bn(pretrained = True)

    # Modify the final fully connected layer for 10 classes
    vgg19_bn.classifier[6] = torch.nn.Linear(4096, 10)
    for param in vgg19_bn.features.parameters():
        param.requires_grad = False

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg19_bn.parameters(), lr=0.001, momentum=0.9)

    torchsummary.summary(vgg19_bn, (3, 32, 32))
    model = vgg19_bn
    train_losses = []  # List to store training losses
    val_losses = []    # List to store validation losses
    train_accuracy = []  # List to store training accuracy
    val_accuracy = []
    for epoch in range(60):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        if epoch == 39 or epoch == 49 or epoch == 59 or epoch ==  60:
            checkpoint_filename = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_filename)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracy, label='Training Accuracy')
            plt.plot(val_accuracy, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()

            plt.show()

        # Calculate and store the validation loss and accuracy
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(trainloader))
        val_losses.append(val_loss / len(testloader))

        train_accuracy.append(100 * correct_train / total_train)
        val_accuracy.append(100 * correct_val / total_val)

    print('Finished Training')

    # Plot the training and validation losses
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()