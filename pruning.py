import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

model = models.squeezenet1_0(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))

model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
])

testset = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=10, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False)

def evaluate(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

accuracy_before = evaluate(model)
print(f"Accuracy before pruning: {accuracy_before:.2f}%")

parameters_to_prune = (
    (model.classifier[1], 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)

# removing pruning reparameterization as I want clean weights after pruning
prune.remove(model.classifier[1], 'weight')

accuracy_after = evaluate(model)
print(f"Accuracy after pruning: {accuracy_after:.2f}%")
