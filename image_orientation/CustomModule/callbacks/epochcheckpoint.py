import os

Count = 0

class EpochCheckpoint(callback):
    @staticmethod
    def on_epoch_end(path, startAt):
        global Count
        if startAt == 0:
            fname = os.path.sep.join([path, "weights-{:03d}.hdf5".format(Count)])
            Count += 1
        else:
            fname = os.path.sep.join([path, "weights-{:03d}.hdf5".format(Count + startAt)])
            Count += 1

        check = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=False, verbose=1)
        return check
