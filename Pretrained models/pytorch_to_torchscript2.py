import torch
import torchvision
import torch.nn as nn
from torchsummary import summary
from cnn_finetune import make_model
#Reference: https://pytorch.org/tutorials/advanced/cpp_export.html

#This function is used to replace the FC layers of vgg16
def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

model = make_model('vgg16', num_classes=4, pretrained=True, input_size=(224, 224), classifier_factory=make_classifier)

#n=0-all layers frozen
#n=4-unfreezing last 4 layers of vgg16 first sequential block
#n=8 - unfreezing last 8 layers of vgg16 first sequential block
total_count = 0
p_count = 0
n = 0 
count = 0
for child in model.children():
    for ch in child.children():
        total_count += 1
    break

for child in model.children():
    p_count += 1

    if p_count == 1:
        for ch in child.children():
            count += 1

            if total_count - count < n:
                for param in ch.parameters():
                    param.requires_grad = True
            else:
                for param in ch.parameters():
                    param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad =  True

#Summary of model alongwith trainable and non trainable params information.
summary(model, (3, 224, 224), device='cpu')

#Set the model in evaluation mode. This is need to ignore dropout and batchnorm operations
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# By tracing it learns the flow of network through example and stores the network architecture.
# It doesn't learn any data dependant information, for which there is another method called scripting.
traced_script_module = torch.jit.trace(model, example)

#We can test the model on sample input and this result can be compare when used in libtorch
output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output[0, :5])

traced_script_module.save("vgg16_model0.pt")