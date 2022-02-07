#include <iostream>
#include <torch/torch.h>

void torch_sample()
{
	const torch::Tensor tensor = torch::randn({ 3,3 });
    std::cout << "The random matrix is:" << '\n' << tensor << '\n';

    // Initialize the device to CPU
    torch::DeviceType device = torch::kCPU;

	// If CUDA is available,run on GPU
    if (torch::cuda::is_available())
    {
        device = torch::kCUDA;
    }

	std::cout << "Running on: " << (device == torch::kCUDA ? "GPU" : "CPU") << '\n';
}