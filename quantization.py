import torch
import torch.nn as nn
import torch.quantization
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

model_fp32 = models.squeezenet1_0(pretrained=True)
model_fp32.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))

model_fp32.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # <-- add this line!
    transforms.ToTensor(),

])

testset = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=10, transform=transform)
testloader = DataLoader(testset, batch_size=16)


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

original_accuracy = evaluate(model_fp32)
print(f"Original model accuracy: {original_accuracy:.2f}%")

model_quantized = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
)

quantized_accuracy = evaluate(model_quantized)
print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")
