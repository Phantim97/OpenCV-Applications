#include <iostream>
#include <string>
#include <fstream>
#include <torch/torch.h>

#include "env_util.h"

//creating an alias so that the code is short.
namespace F = torch::nn::functional;

// Where to find the CIFAR10 dataset.
const std::string k_data_root = util::get_dataset_path() + "cifar-10-batches-bin";

// The batch size for training.
constexpr int64_t k_train_batch_size = 64;

// The batch size for testing.
constexpr int64_t k_test_batch_size = 1000;

// The number of epochs to train.
constexpr int64_t k_number_of_epochs = 5;

// After how many batches to log a new update with the loss value.
constexpr int64_t k_log_interval = 10;

constexpr uint32_t k_image_rows = 32;
constexpr uint32_t k_image_columns = 32;
constexpr uint32_t k_image_channels = 3;
constexpr uint32_t k_num_train_batch_files = 5;
const std::string k_train_batch_file_names[] = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
constexpr uint32_t k_train_size_per_batch_file = 10000;
constexpr uint32_t k_train_size = 50000;
constexpr uint32_t k_test_size = 10000;
const std::string k_test_filename = "test_batch.bin";

/// The CIFAR10 dataset.
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10>
{
public:
	/// The mode in which the dataset is loaded.
	enum class Mode { k_train, k_test };
	/// Loads the CIFAR dataset from the `root` path.
	///
	/// The supplied `root` path should contain the *content* of the unzipped
	/// CIFAR binary version dataset, available from https://wwww.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz.
	explicit CIFAR10(const std::string& root, Mode mode = Mode::k_train)
	{
		char label;
		torch::Tensor image = torch::empty({ k_image_channels, k_image_rows, k_image_columns }, torch::kByte);
		if (Mode::k_train == mode)
		{
			targets_ = torch::empty(k_train_size, torch::kByte);
			images_ = torch::empty({ k_train_size, k_image_channels, k_image_rows, k_image_columns }, torch::kByte);

			size_t sample_index = 0;

			for (size_t batch = 0; batch < k_num_train_batch_files; ++batch) 
			{
				std::string path = join_paths(root, k_train_batch_file_names[batch]);
				std::ifstream fid(path, std::ios::binary);
				TORCH_CHECK(fid, "Error opening images file at ", path)

				for (size_t img_index = 0; img_index < k_train_size_per_batch_file; ++img_index)
				{
					fid.read(&label, sizeof(label));
					targets_[sample_index] = label;
					fid.read(static_cast<char*>(image.data_ptr()), image.numel());
					images_[sample_index] = image.clone();
					sample_index = sample_index + 1;
				}

				fid.close();
			}
		}
		else 
		{
			targets_ = torch::empty(k_test_size, torch::kByte);
			images_ = torch::empty({ k_test_size, k_image_channels, k_image_rows, k_image_columns }, torch::kByte);

			size_t sample_index = 0;

			std::string path = join_paths(root, k_test_filename);
			std::ifstream fid(path, std::ios::binary);
			TORCH_CHECK(fid, "Error opening images file at ", path)

			for (size_t img_index = 0; img_index < k_test_size; ++img_index)
			{
				fid.read(&label, sizeof(label));
				targets_[sample_index] = label;

				fid.read(static_cast<char*>(image.data_ptr()), image.numel());
				images_[sample_index] = image.clone();
				sample_index = sample_index + 1;
			}

			fid.close();
		}

		images_
		= images_.to(torch::kFloat32).div_(255);
		targets_ = targets_.to(torch::kInt64);
	}

	/// Returns the `Example` at the given `index`.
	torch::data::Example<> get(size_t index)
	{
		return { images_[index], targets_[index] };
	}

	/// Returns the size of the dataset.
	torch::optional<size_t> size() const override
	{
		return images_.size(0);
	}

	/// Returns true if this is the training subset of MNIST.
	bool is_train() const noexcept
	{
		return images_.size(0) == k_train_size;
	}

	/// Returns all images stacked into a single tensor.
	const torch::Tensor& images() const
	{
		return images_;
	}

	/// Returns all targets stacked into a single tensor.
	const torch::Tensor& targets() const
	{
		return targets_;
	}

private:
	torch::Tensor images_, targets_;

	static std::string join_paths(std::string head, const std::string& tail)
	{
		if (head.back() != '/')
		{
			head.push_back('/');
		}

		head += tail;
		return head;
	}
};


//////////// Specify the architecture.

//Model that worked
//struct Net : torch::nn::Module
//{
//	Net()
//	{
//		conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
//		conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3)));
//		dp1 = register_module("dp1", torch::nn::Dropout(0.5));
//		conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
//		conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
//		dp2 = register_module("dp2", torch::nn::Dropout(0.5));
//		conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
//		conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
//		dp3 = register_module("dp3", torch::nn::Dropout(0.5));
//		fc1 = register_module("fc1", torch::nn::Linear(2 * 2 * 64, 512));
//		dp4 = register_module("dp4", torch::nn::Dropout(0.40));
//		fc2 = register_module("fc2", torch::nn::Linear(512, 10));
//	}
//
//	torch::Tensor forward(torch::Tensor x)
//	{
//		x = torch::tanh(conv1_1->forward(x));
//		x = torch::tanh(conv1_2->forward(x));
//		x = torch::max_pool2d(x, 2);
//		x = dp1(x);
//
//		x = torch::tanh(conv2_1->forward(x));
//		x = torch::tanh(conv2_2->forward(x));
//		x = torch::max_pool2d(x, 2);
//		x = dp2(x);
//
//		x = torch::tanh(conv3_1->forward(x));
//		x = torch::tanh(conv3_2->forward(x));
//		x = torch::max_pool2d(x, 2);
//		x = dp3(x);
//
//		x = x.view({ -1, 2 * 2 * 64 });
//
//		x = torch::tanh(fc1->forward(x));
//		x = dp4(x);
//		x = torch::log_softmax(fc2->forward(x), 1);
//
//		return x;
//	}
//
//	torch::nn::Conv2d conv1_1{ nullptr };
//	torch::nn::Conv2d conv1_2{ nullptr };
//	torch::nn::Conv2d conv2_1{ nullptr };
//	torch::nn::Conv2d conv2_2{ nullptr };
//	torch::nn::Conv2d conv3_1{ nullptr };
//	torch::nn::Conv2d conv3_2{ nullptr };
//	torch::nn::Dropout dp1{ nullptr };
//	torch::nn::Dropout dp2{ nullptr };
//	torch::nn::Dropout dp3{ nullptr };
//	torch::nn::Dropout dp4{ nullptr };
//	torch::nn::Linear fc1{ nullptr };
//	torch::nn::Linear fc2{ nullptr };
//};

struct Net final : torch::nn::Module
{
	Net()
	{
		//torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(p).stride(s) and similary other options
		conv1_1 = register_module("conv1_1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(3, 128, 5).padding(1)));
		conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 5)));
		dp1 = register_module("dp1", torch::nn::Dropout(0.5));
		conv2_1 = register_module("conv2_1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 5).padding(1)));
		conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
		dp2 = register_module("dp2", torch::nn::Dropout(0.5));
		conv3_1 = register_module("conv3_1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 3).padding(1)));
		conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3)));
		dp3 = register_module("dp3", torch::nn::Dropout(0.5));
		fc1 = register_module("fc1", torch::nn::Linear(2 * 2 * 32, 512));
		dp4 = register_module("dp4", torch::nn::Dropout(0.4));
		fc2 = register_module("fc2", torch::nn::Linear(512, 10));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::tanh(conv1_1->forward(x));
		x = torch::tanh(conv1_2->forward(x));
		x = torch::max_pool2d(x, 2);
		x = dp1(x);

		x = torch::tanh(conv2_1->forward(x));
		x = torch::tanh(conv2_2->forward(x));
		x = torch::max_pool2d(x, 2);
		x = dp2(x);

		x = torch::tanh(conv3_1->forward(x));
		x = torch::tanh(conv3_2->forward(x));
		x = torch::max_pool2d(x, 2);
		x = dp3(x);

		x = x.view({ -1, 2 * 2 * 32 });

		x = torch::tanh(fc1->forward(x));
		x = dp4(x);
		x = torch::log_softmax(fc2->forward(x), 1);

		return x;
	}

	torch::nn::ConvTranspose2d conv1_1{ nullptr };
	torch::nn::Conv2d conv1_2{ nullptr };
	torch::nn::ConvTranspose2d conv2_1{ nullptr };
	torch::nn::Conv2d conv2_2{ nullptr };
	torch::nn::ConvTranspose2d conv3_1{ nullptr };
	torch::nn::Conv2d conv3_2{ nullptr };
	torch::nn::Dropout dp1{ nullptr };
	torch::nn::Dropout dp2{ nullptr };
	torch::nn::Dropout dp3{ nullptr };
	torch::nn::Dropout dp4{ nullptr };
	//torch::nn::Dropout dp5{ nullptr };
	torch::nn::Linear fc1{ nullptr };
	torch::nn::Linear fc2{ nullptr };
	//torch::nn::Linear fc3{ nullptr };
};

////////////////////////////////////////////////////////////

template <typename DataLoader>
void train(const int32_t epoch, Net& model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, const size_t dataset_size)
{
	model.train();
	double train_loss = 0;
	int32_t correct = 0;
	size_t batch_idx = 0;

	for (torch::data::Example<>& batch : data_loader)
	{
		const torch::Tensor data = batch.data.to(device);
		torch::Tensor targets = batch.target.to(device);
		optimizer.zero_grad();
		torch::Tensor output = model.forward(data);
		torch::Tensor loss = F::nll_loss(output, targets);
		AT_ASSERT(!std::isnan(loss.item<float>()));
		loss.backward();
		optimizer.step();

		if (batch_idx++ % k_log_interval == 0)
		{
			std::printf(
				"\rTrain Epoch: %d [%5ld/%5ld] Loss: %.4f",
				epoch,
				batch_idx * batch.data.size(0),
				dataset_size,
				loss.item<float>());
		}
		train_loss += loss.item<float>();
		torch::Tensor pred = output.argmax(1);
		correct += pred.eq(targets).sum().item<int64_t>();
	}

	train_loss /= dataset_size;
	std::printf(
		"\n   Train set: Average loss: %.4f | Accuracy: %.3f",
		train_loss,
		static_cast<double>(correct) / dataset_size);
}

template <typename DataLoader>
void test(Net& model, torch::Device device, DataLoader& data_loader, size_t dataset_size)
{
	torch::NoGradGuard no_grad;
	model.eval();
	double test_loss = 0;
	int32_t correct = 0;
	for (const torch::data::Example<>& batch : data_loader)
	{
		const torch::Tensor data = batch.data.to(device);
		torch::Tensor targets = batch.target.to(device);
		torch::Tensor output = model.forward(data);
		test_loss += F::nll_loss(
			output,
			targets,
			F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kSum))
			.item<float>();
		torch::Tensor pred = output.argmax(1);
		correct += pred.eq(targets).sum().item<int64_t>();
	}

	test_loss /= dataset_size;
	std::printf(
		"\n    Test set: Average loss: %.4f | Accuracy: %.3f\n",
		test_loss,
		static_cast<double>(correct) / dataset_size);
}

int main()
{
	torch::manual_seed(1);

	torch::DeviceType device_type;

	if (torch::cuda::is_available())
	{
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else 
	{
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}

	const torch::Device device(device_type);

	Net model;
	model.to(device);

	torch::data::datasets::MapDataset<CIFAR10, torch::data::transforms::Stack<>> train_dataset =
		CIFAR10(k_data_root, CIFAR10::Mode::k_train).map(torch::data::transforms::Stack<>());

	const size_t train_dataset_size = train_dataset.size().value();

	const std::unique_ptr<torch::data::StatelessDataLoader<
		torch::data::datasets::MapDataset<CIFAR10, torch::data::transforms::Stack<>>,
		torch::data::samplers::SequentialSampler>> train_loader = torch::data::make_data_loader<
		torch::data::samplers::SequentialSampler>(std::move(train_dataset), k_train_batch_size);

	torch::data::datasets::MapDataset<CIFAR10, torch::data::transforms::Stack<>> test_dataset =
		CIFAR10(k_data_root, CIFAR10::Mode::k_test).map(torch::data::transforms::Stack<>());

	const size_t test_dataset_size = test_dataset.size().value();

	const std::unique_ptr<torch::data::StatelessDataLoader<
		torch::data::datasets::MapDataset<CIFAR10, torch::data::transforms::Stack<>>,
		torch::data::samplers::RandomSampler>> test_loader = torch::data::make_data_loader(
		std::move(test_dataset), k_test_batch_size);

	// You must specify the kind of optimizer in the following code

	//torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(/*lr=*/0.001).momentum(0.9));
	torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0007).amsgrad(true));
	//torch::optim::RMSprop optimizer(model.parameters(), torch::optim::RMSpropOptions(0.001));
	////////////////////////////////////////////////////////////////

	for (size_t epoch = 1; epoch <= k_number_of_epochs; ++epoch)
	{
		train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
		test(model, device, *test_loader, test_dataset_size);
	}

	return 0;
}

void disp(const torch::Tensor& t)
{
	std::cout << t << '\n';
}

void size(const torch::Tensor& t)
{
	std::cout << t.sizes() << '\n';
}