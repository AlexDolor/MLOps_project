import torchvision

if __name__ == '__main__':
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True)
    savedir = "Homework\\testing\\images\\"
    n = 1000
    for i, (img, label) in enumerate(testset):
        # print(type(img), type(label))
        img.save(savedir + f'{i}_{label}.png')
        if i >= n: break


