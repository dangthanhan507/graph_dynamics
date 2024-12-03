import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_losses = np.load('chkpt/train_losses.npy')
    val_losses = np.load('chkpt/validation_losses.npy')
    
    epoch = np.arange(len(train_losses))
    
    plt.plot(epoch, train_losses, c='r',label='train')
    plt.plot(epoch, val_losses, c='g', label='val')
    #ony show integer ticks
    plt.xticks(epoch[::10])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (mm)')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()