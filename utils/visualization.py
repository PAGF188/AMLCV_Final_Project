import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def labeled_face_visualization(face, gender, landmarks, ax = None):
    """
    Based on the teacher's code
    """
    lm_marker = ['<', '>', '^', '3', '4']
    gender_txt = ['Woman', 'Man']

    if ax is None:
      ax = plt.gca()

    ax.imshow(np.asarray(face).mean(2), cmap='gray')
    ax.title.set_text(gender_txt[gender.item()])

    lm = landmarks.reshape(-1, 2)
    for i, m in enumerate(lm_marker):
      ax. scatter(lm[i,0], lm[i,1], marker=lm_marker[i])

def plot_dataset_examples(dataset):
    """
    Based on the teacher's code
    
    Parameters
    ----------
    dataset : CelebAMini(VisionDataset)
        Data
    """
    # We randomize an index to select some few random samples for visualization
    view = random.permutation(len(dataset))

    # Subplots:
    rows = 2
    cols = 5
    size_each = 3

    fig, axes = plt.subplots(rows*2, cols,figsize =(cols*size_each, rows*2*size_each) )

    # Remove the plot axis
    for ax in axes.reshape(-1):
        ax.axis('off')

    counter = 0
    for r in range(rows):
        for c in range(cols):
            face, (gender, landmarks) = dataset[view[counter]]
            counter += 1

            axes[r*2][c].imshow(face)
            labeled_face_visualization(face, gender, landmarks, axes[r*2+1][c])
    plt.show()