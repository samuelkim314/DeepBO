import os
from os.path import isfile, join, exists


class GenericFile:
    """File manager for writing training/validation loss"""
    def __init__(self, path, name='loss', continue_file=False):
        if not exists(path):
            os.makedirs(path)
        i_file = 0
        while True:
            file_name = join(path, name + str(i_file) + ".txt")
            if isfile(file_name):
                i_file += 1
            else:
                if continue_file:
                    i_file -= 1
                    file_name = join(path, name + str(i_file) + ".txt")
                break
        if continue_file:
            mode = 'a'
        else:
            mode = 'w'
        self.loss_file = open(file_name, mode)

    def write_loss(self, loss_str):
        self.loss_file.write(str(loss_str) + "\n")
        self.loss_file.flush()

    def close(self):
        self.loss_file.close()


class LossFile:
    """File manager for writing training/validation loss"""
    def __init__(self, path, continue_file=False):
        if not exists(path):
            os.makedirs(path)
        i_file = 0
        while True:
            train_file_name = join(path, "trainLoss_" + str(i_file) + ".txt")
            if isfile(train_file_name):
                i_file += 1
            else:
                if continue_file:
                    i_file -= 1
                    train_file_name = join(path, "trainLoss_" + str(i_file) + ".txt")
                break
        if continue_file:
            mode = 'a'
        else:
            mode = 'w'
        self.train_loss_file = open(train_file_name, mode)
        self.val_loss_file = open(join(path, "valLoss_" + str(i_file) + ".txt"), mode)

    def write_loss(self, train_loss, val_loss):
        self.train_loss_file.write(str(train_loss) + "\n")
        self.val_loss_file.write(str(val_loss) + "\n")
        self.train_loss_file.flush()
        self.val_loss_file.flush()

    def close(self):
        self.train_loss_file.close()
        self.val_loss_file.close()

