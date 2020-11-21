import torch
import torch.nn as nn
from torch.autograd import Variable
import dataset
import one_hot_encoding as ohe
from model import CNN
import settings
import numpy as np

# Hyper Parameters
epochs = 10
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels.float()).to(device)
        predict_labels = model(images)
        # print(predict_labels.type, predict_labels)
        # print(labels.type, predict_labels)

        loss = criterion(predict_labels, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("epoch:", epoch, "step:", i, "loss:", loss.item())

    torch.save(model.state_dict(), f"./model-{epoch}.pkl")


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.MultiLabelSoftMarginLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            output = model(images)

            test_loss += criterion(output, labels).item()
            c0 = settings.CHAR_SET[
                np.argmax(output[0, 0 : settings.CHAR_SET_LEN].data.cpu().numpy())
            ]
            c1 = settings.CHAR_SET[
                np.argmax(
                    output[
                        0, settings.CHAR_SET_LEN : 2 * settings.CHAR_SET_LEN,
                    ].data.cpu().numpy()
                )
            ]
            predict_label = "%s%s" % (c0, c1)
            true_label = ohe.decode(labels.cpu().numpy()[0])

            if predict_label == true_label:
                correct += 1

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)
    return test_loss, test_acc


def main():
    model = CNN().to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.ExponentialLR(optimizer, gamma=0.9)

    train_loader = dataset.get_train_data_loader()
    validation_loader = dataset.get_validation_data_loader()
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, epoch)
        scheduler.step()
        test_loss, test_accuracy = evaluate(model, validation_loader)
        print(f"[{epoch}] Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), "./model.pkl")


if __name__ == "__main__":
    main()
