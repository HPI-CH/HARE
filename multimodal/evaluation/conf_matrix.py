from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


def create_conf_matrix(path: str, y_test_pred: np.ndarray, y_test_true: np.ndarray, file_name: str = "conf_matrix", title="Confusion Matrix", label_mapping: dict = None) -> None:
    """
    creates and saves conf matrix as .png to path 

    TODO: this function shows a window on your desktop, not needed its saved in the folder, deactivate that
    (PS: its not imshow, haha)
    """
    # reset stupid singleton
    plt.cla()
    plt.clf()

    # Convertion
    classes = list(range(len(y_test_true[0]))) # [0, 1, 0, 0, 0] to [0, 1, 2, 3, 4]
    y_test_true = np.argmax(y_test_true, axis=1)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    if label_mapping != None:
        y_test_true = [label_mapping[int(l)] for l in y_test_true]
        y_test_pred = [label_mapping[int(l)] for l in y_test_pred]
        classes = [label_mapping[int(c)] for c in classes]

    # Settings
    cm = confusion_matrix(y_test_true, y_test_pred, labels= classes)
    normalize = False
    cmap = plt.cm.Blues
    
    # Create Conf-Plot
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    f3 = plt.figure(3, figsize=(len(classes)*0.6, len(classes)*0.7))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("predicted label")
        f3.show()

    # Save
    plt.savefig(os.path.join(path, f"{file_name}.png"))