import torch
import torchvision

#Reference: https://pytorch.org/tutorials/advanced/cpp_export.html

# An instance of your model.
model = torchvision.models.mobilenet_v2(pretrained=True)

#Similarly we can load inception_v3, vgg16, resnet50
#torchvision.models.inception_v3(pretrained=True)
#torchvision.models.vgg16(pretrained=True)
#torchvision.models.resnet50(pretrained=True)
#And change the model file name below accordingly.

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

traced_script_module.save("mobilenet_v2_model.pt")