import os
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

from ViTinyBase import ViTinyBase

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

BATCH_SIZE = 5
EPOCHS = 2

# normalizes images from [0,1] to [-1, 1]
normalize_images = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def imshow(img, label=None):
    img = img / 2 + 0.5
    img = rearrange(img.numpy(), "c w h -> w h c")
    plt.imshow(img)
    if label:
        plt.title(label)
    plt.show()


if __name__ == '__main__':
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=normalize_images)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=normalize_images)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_iter = iter(train_loader)
    images, labels = next(train_iter)

    labels_str = '[' + ', '.join([classes[lbl] for lbl in labels]) + ']'

    imshow(torchvision.utils.make_grid(images), labels_str)

    image_size = images.size()[-2:]

    # split image into 4x4 patches
    patch_size = (dim // 4 for dim in image_size)

    # cuda0 = torch.device('cuda:0')  # CUDA GPU 0

    model = ViTinyBase(image_size, patch_size, len(classes), 5, 16, 16)
    # model.to(cuda0)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # todo: test NAdam

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # inputs = inputs.to(cuda0)
            # labels = labels.to(cuda0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), os.path.join(MODELS_FOLDER, 'cifar_vitiny.pth'))
    print('Finished Training')
