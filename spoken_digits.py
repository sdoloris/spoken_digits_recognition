import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random
import numpy as np
from model import Net

# Custom dataset for use of dataloader
class AudioSampleDataset(Dataset):
    """ Audio samples as MFCC features dataset """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return {'X': self.X[idx], 'y': self.y[idx]}


def evaluate_model(net, X_test, y_test):
    """ Evaluate the model's performance on the test set X_test and y_test
    
    Parameters:
    ----------
    X_test: torch tensor of MFCC
    y_test: torch tensor of labels

    Returns:
    -------
    accuracy: average accuracy of the model
    accuracies: array of accuracy for each class
    """
    accuracy = 0 # average accuracy of the model
    accuracies = np.zeros(11) # accuracy for each class
    nb_occurences = np.zeros(11) # to convert counts to accuracy

    with torch.no_grad():
        for i in range(len(X_test)):
            prediction = net.predict(X_test[i].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float))
            label = (int) (y_test[i].to(device=device).item())
            if prediction == label:
                accuracies[label] = accuracies[label] + 1
                accuracy = accuracy + 1
            nb_occurences[label] = nb_occurences[label] + 1

    accuracy = accuracy/len(X_test)
    accuracies = np.divide(accuracies, nb_occurences)

    return accuracy, accuracies

if __name__ == "__main__":
    print("Initializing the neural network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    print("Loading the data...")
    X = np.load("X.npy")
    y = np.load("y.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    train_dataset = AudioSampleDataset(X_train, y_train)
    dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)

    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    best_accuracy = 0.0
    print("Started training !")
    for epoch in range(201):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data['X'].unsqueeze(1).float().to(device), data['y'].long().to(device)

            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print('[epoch %d] loss: %.3f' % (epoch, running_loss))
        # Evaluate the model on the test set every 10 epochs
        if epoch % 10 == 0:
            accuracy, accuracies = evaluate_model(net, X_test, y_test)
            print('accuracy', accuracy)
            print('accuracies', accuracies)
            # Saving model if accuracy on the test set is better than previous best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), 'best_model_state.pt')
                print("Best model saved")