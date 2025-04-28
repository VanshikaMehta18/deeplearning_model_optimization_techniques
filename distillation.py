import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

teacher = models.squeezenet1_0(pretrained=True)
teacher.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
teacher.eval()

student = models.squeezenet1_0(pretrained=False)
student.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# preparing FakeData
trainset = datasets.FakeData(size=200, image_size=(3, 224, 224), num_classes=10, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

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

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = self.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        hard_loss = self.ce_loss(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher.to(device)
student.to(device)

criterion = DistillationLoss(temperature=2.0, alpha=0.7)
optimizer = optim.Adam(student.parameters(), lr=0.001)

# training the student
num_epochs = 5
for epoch in range(num_epochs):
    student.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)

        student_outputs = student(inputs)

        loss = criterion(student_outputs, teacher_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")

teacher_acc = evaluate(teacher)
student_acc = evaluate(student)
print(f"Teacher accuracy: {teacher_acc:.2f}%")
print(f"Student accuracy after distillation: {student_acc:.2f}%")
