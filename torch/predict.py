import numpy as np
import torch
from torch.autograd import Variable
import settings
import dataset
from model import CNN

def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model-6.pkl'))

    test_loader = dataset.get_test_data_loader()
    correct = 0
    for i, (image, label) in enumerate(test_loader):
        image = Variable(image)
        predicted = cnn(image)
        answer = label.numpy()[0]

        c0 = settings.CHAR_SET[np.argmax(predicted[0, 0:settings.CHAR_SET_LEN].data.numpy())]
        c1 = settings.CHAR_SET[np.argmax(predicted[0, settings.CHAR_SET_LEN:2 * settings.CHAR_SET_LEN].data.numpy())]

        a0 = settings.CHAR_SET[np.argmax(answer[0:settings.CHAR_SET_LEN])]
        a1 = settings.CHAR_SET[np.argmax(answer[settings.CHAR_SET_LEN:2 * settings.CHAR_SET_LEN])]
        predicted_num = '%s%s' % (c0, c1)
        answer_num = '%s%s' % (a0, a1)

        if predicted_num == answer_num:
            correct += 1
        # print(predicted_num == answer_num, predicted_num, answer_num)
    
    test_acc = 100.0 * correct / len(test_loader.dataset)
    print(f"[*] Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc} %)")

if __name__ == '__main__':
    main()
