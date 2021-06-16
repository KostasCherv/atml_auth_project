import torch
from argparse import ArgumentParser
from torchvision import models, datasets, transforms
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from PIL import Image
from lime import *
import json
from skimage.segmentation import mark_boundaries
from lime import lime_image
import torch.nn.functional as F
from rise.evaluation import CausalMetric, auc, gkern
from rise.explanations import RISE
import shap
from nn_interpretability.interpretation.deeplift.deeplift import DeepLIFT, DeepLIFTRules


model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf


def batch_predict(images):
    global device
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['vgg11', 'vgg19'], default='vgg11')
    parser.add_argument('--dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--batch-size', type=int, default='10')
    parser.add_argument('--epochs', type=int, default='10')
    parser.add_argument('--samples', type=int, default='10')
    parser.add_argument('--test', action='store_true')

    return parser.parse_args()

def test(net, testloader):
    print('Testing')
    global device
    correct = 0
    total = 0
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def rise_test(img):
    global model
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        klen = 11
        ksig = 5
        kern = gkern(klen, ksig)

        # Function that blurs input image
        blur = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen // 2)

        explainer = RISE(model, (224, 224))
        explainer.generate_masks(N=5000, s=10, p1=0.1)
        img = img.convert('RGB')
        img_t = get_input_tensors(img).to(device)
        sal = explainer(img_t.cuda())[1].cpu().numpy()
        return sal
        #plt.imshow(sal, cmap='jet', alpha=0.5)



def deepLift_test(img, class_id):
    global device, model
    img_t = get_input_tensors(img).to(device)
    model = model.to(device)
    model.eval()
    interpreter = DeepLIFT(model, [class_id], None, DeepLIFTRules.NoRule)
    out = interpreter.interpret(img_t)
    out = (out - out.min()) / (out.max() - out.min())
    out = transformToPil(out.squeeze(0))
    return out


def lime_test(img, class_labels):
    global model, device
    #test = transformToPil(img)
    img = img.convert('RGB')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        img_t = get_input_tensors(img).to(device)
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)
        probs5 = probs.cpu().topk(5)
        predictions = tuple((p, c, class_labels[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))

        test_pred = batch_predict([pill_transf(img)])
        test_pred.squeeze().argmax()
        print(test_pred)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                                 batch_predict,  # classification function
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)  # number of images that will be sent to classification function

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=False)
        img_boundry1 = mark_boundaries(temp / 255.0, mask)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)
        img_boundry2 = mark_boundaries(temp / 255.0, mask)

        return img_boundry2


def plot(img, lime, rise, coda, label):

    fig, (im1, im2, im3, im4) = plt.subplots(1,4)
    fig.suptitle(label)
    im1.imshow(img)
    im2.imshow(lime)
    im3.imshow(rise)
    im4.imshow(coda)
    im1.set_title('Input')
    im2.set_title('LIME')
    im3.set_title('RISE')
    im4.set_title('DeepLift')
    im1.axis('off')
    im2.axis('off')
    im3.axis('off')
    im4.axis('off')
    plt.show()




def train_epoch(net, trainloader, criterion, optimizer, path, epoch):
    global device
    print('Training epoch:', epoch+1)
    epoch_loss = 0.0
    net = net.to(device)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()
    print('Epoch loss: %.3f' % (epoch_loss/len(trainloader)))
    torch.save(net.state_dict(), path)

transformToPil = transforms.ToPILImage()
transformToTensor = transforms.ToTensor()
pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

if __name__ == '__main__':
    print('PyTorch Version:', torch.__version__)
    #global model
    args = parseArguments()
    modelName = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    samples = args.samples
    epochs = args.epochs
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'cifar10':
        data = datasets.cifar.CIFAR10(root='./data', download=True)
        numClasses = 10
        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)

    if modelName == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=numClasses)
    elif model == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=numClasses)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_path = './{}_{}.pth'.format(dataset, modelName)

    if not os.path.isfile(model_path):
        for epoch in range(epochs):
            train_epoch(net=net, trainloader=trainloader, path=model_path, criterion=criterion, optimizer=optimizer,
                        epoch=epoch, device=device)
    else:
        print('Loading model...')
        model.load_state_dict(torch.load(model_path))

    if args.test:
        test(testloader=testloader)

    id_list = np.random.randint(len(data), size=samples)
    for id in id_list:
        (img, label) = data[id]
        lime = lime_test(img=img, class_labels=classes)
        rise = rise_test(img=img)
        deepLift = deepLift_test(img, label)
        plot(img, lime, rise, deepLift, classes[label])







