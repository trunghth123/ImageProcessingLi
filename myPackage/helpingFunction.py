import matplotlib.pyplot as plt


def showImage(myImage):
    if (myImage.ndim>2):
        myImage = myImage[:,:,::-1] #OpenCV follows BGR, matplotlib likely follows RGB

    fig = plt.subplot()

    fig.imshow(myImage, cmap = 'gray', interpolation = 'bicubic')

    plt.show()
    return fig