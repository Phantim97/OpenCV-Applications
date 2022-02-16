#include <stdint.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "env_util.h"

//Read mnist
static uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/*We can find the mnist file foramt at http://yann.lecun.com/exdb/mnist/
Ubyte file consists values like MagicNumber, NumItems, NumRows, NumCols, Data
Here, MagicNumber is unique to type like images or labels */
static torch::Tensor read_mnist_images(const std::string& image_filename)
{
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);

    //2051 is magic number for images
    if (magic != 2051)
    {
        std::cout << "Incorrect image file magic: " << magic << '\n';
        return torch::tensor(-1);
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    image_file.read(reinterpret_cast<char*>(&cols), 4);

    const torch::Tensor tensor = torch::empty({num_items, 1, rows, cols}, torch::kByte);
    image_file.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());

    return tensor.to(torch::kFloat32).div_(255);
}

static torch::Tensor read_mnist_labels(const std::string& label_filename)
{
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    uint32_t magic;
    uint32_t num_labels;

    label_file.read(reinterpret_cast<char*>(&magic), 4);

    //2049 is magic number for images
    if (magic != 2049) 
    {
        std::cout << "Incorrect image file magic: " << magic << '\n';
        return torch::tensor(-1);
    }

    label_file.read(reinterpret_cast<char*>(&num_labels), 4);

    const torch::Tensor tensor = torch::empty(num_labels, torch::kByte);
    label_file.read(reinterpret_cast<char*>(tensor.data_ptr()), num_labels);
    return tensor.to(torch::kInt64);
}

//Classification portion
struct Options
{
    int batch_size = 100; //Batch size
    size_t epochs = 20; // Number of epochs
    size_t log_interval = 20;
    std::ofstream loss_acc_train;
    std::ofstream loss_acc_test;
    //Paths to train and test images and labels
    const std::string train_images_path = "train-images-idx3-ubyte";
    const std::string train_labels_path = "train-labels-idx1-ubyte";
    const std::string test_images_path = "t10k-images-idx3-ubyte";
    const std::string test_labels_path = "t10k-labels-idx1-ubyte";
    torch::DeviceType device = torch::kCUDA;
};

static Options options;

//Feed Forward network
struct Net : torch::nn::Module
{
    Net()
	{
        fc1 = register_module("fc1", torch::nn::Linear(28 * 28, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, 10));
    }

    // Implement Forward Pass Algorithm
    torch::Tensor forward(torch::Tensor x)
	{
        x = x.view({ options.batch_size, -1 });
        //Input -> Linear -> Relu -> Linear -> Relu -> Linear -> Softmax Classifier-> Output
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return torch::log_softmax(x, 1);
    }

    //Initilaize the constructor with null pointer. More details given in the reference
    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear fc3{ nullptr };
};

/*Read images from ubyte format and convert to tensors*/
static torch::Tensor process_images(const std::string& root, const bool train)
{
    std::string path = root;
	path.append(train ? options.train_images_path : options.test_images_path); //images_path
    torch::Tensor images = read_mnist_images(path); //refer to read-mnist.h

    return images;
}

/*Read labels from ubyte format and convert to tensors*/
static torch::Tensor process_labels(const std::string& root, const bool train)
{
    std::string path = root;
	path.append(train ? options.train_labels_path : options.test_labels_path); //labels_path
    torch::Tensor labels = read_mnist_labels(path);//refer to read-mnist.h

    return labels;
}

/*Use CustomDataset class to load any type of dataset other than inbuilt datasets*/
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
private:
    /* data */
    // Should be 2 tensors
    torch::Tensor images_;
    torch::Tensor labels_;
    size_t img_size_;
public:
    CustomDataset(const std::string& root, const bool train)
	{
        images_ = process_images(root, train);
        labels_ = process_labels(root, train);
        img_size_ = images_.size(0);
    }

    /*Returns the data sample at the given `index*/
    torch::data::Example<> get(size_t index) override
	{
        /* This should return {torch::Tensor, torch::Tensor} */
        const torch::Tensor img = images_[index];
        const torch::Tensor label = labels_[index];
        return { img.clone(), label.clone() };
    }

    torch::optional<size_t> size() const override
	{
        return img_size_;
    }
};

template <typename DataLoader>
void train(const std::shared_ptr<Net>& network, DataLoader& loader, torch::optim::Optimizer& optimizer, const size_t epoch, const size_t data_size)
{
    size_t index = 0;
    /*Set network in the training mode*/
    network->train();
    float Loss = 0;
    float Acc = 0;

    for (torch::data::Example<>& batch : loader)
    {
	    const torch::Tensor data = batch.data.to(options.device);
        torch::Tensor targets = batch.target.to(options.device).view({ -1 });
        // Execute the model on the input data
        torch::Tensor output = network->forward(data);

        //Using mean square error loss function to compute loss
        torch::Tensor loss = torch::nll_loss(output, targets);
        torch::Tensor acc = output.argmax(1).eq(targets).sum();

        // Reset gradients
        optimizer.zero_grad();
        // Compute gradients
        loss.backward();
        //Update the parameters
        optimizer.step();

        Loss += loss.item<float>();
        Acc += acc.item<float>();
    }

    if (index++ % options.log_interval == 0)
    {
	    const size_t end = data_size;

        std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
            << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end << '\n';
    }
}

template <typename DataLoader>
void test(const std::shared_ptr<Net>& network, DataLoader& loader, const size_t epoch, const size_t data_size)
{
    network->eval();
    size_t index = 0;
    float Loss = 0, Acc = 0;
    int display_count = 0;

    for (const torch::data::Example<>& batch : loader)
    {
        torch::Tensor data = batch.data.to(options.device);
        torch::Tensor targets = batch.target.to(options.device).view({-1});

        torch::Tensor output = network->forward(data);

        //To display 3 test image and its output
        if (display_count < 3 && epoch == options.epochs) 
        {
            cv::Mat test_image(28, 28, CV_8UC1);
            torch::Tensor tensor = data[display_count].mul_(255).clamp(0, 255).to(torch::kU8);
            tensor = tensor.to(options.device);
            std::memcpy((void*)test_image.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

            std::cout << "***** TESTING on TEST IMAGE " << display_count << " *****\n";
            std::cout << "GroundTruth: " << targets[display_count].template item<float>()
                << ", Prediction: " << output[display_count].argmax() << '\n';
            std::cout << "Output Probabilities\n";

            for (int i = 0; i < output[display_count].size(0); i++) 
            {
                std::cout << "Class: " << i << " " << torch::exp(output[display_count])[i].template item<float>() << '\n';
            }

            cv::imwrite("OUTPUT_GT_" + std::to_string(targets[display_count].template item<int>()) +
                "_Pred_" + std::to_string(output[display_count].argmax().template item<int>()) + ".jpg", test_image);
            std::cout << "Outputs saved, Please checkout the output images\n";

            display_count++;
        }

        torch::Tensor loss = torch::nll_loss(output, targets);
        torch::Tensor acc = output.argmax(1).eq(targets).sum();

        Loss += loss.item<float>();
        Acc += acc.item<float>();
    }

    if (index++ % options.log_interval == 0)
    {
        options.loss_acc_test << std::to_string(Loss / data_size) + "," + std::to_string(Acc / data_size) << std::endl;
        std::cout << "Val Epoch: " << epoch
            << "\tVal Loss: " << Loss / data_size << "\tVal ACC:" << Acc / data_size << '\n';
    }
}

void classification_main()
{
    /*Path to Fashion Mnist*/
    const std::string root_string = util::get_dataset_path() + "fashion-mnist/";
    constexpr bool is_train = true; //Flag to create train or test data

    /*Uses Custom Dataset Class to load train data. Apply stack collation which takes
      batch of tensors and stacks them into single tensor along the first dimension*/
    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>> train_dataset =
	    CustomDataset(root_string, is_train).map(torch::data::transforms::Stack<>());
    /*Data Loader provides options to speed up the data loading like batch size, number of workers*/

    const std::unique_ptr<torch::data::StatelessDataLoader<
	    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>,
	    torch::data::samplers::RandomSampler>> train_loader = torch::data::make_data_loader<
	    torch::data::samplers::RandomSampler>(
	    std::move(train_dataset), options.batch_size);

    const unsigned long long train_size = train_dataset.size().value();

    /*Process and load test dat similar to above*/
    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>> test_dataset =
	    CustomDataset(root_string, false).map(torch::data::transforms::Stack<>());

    const std::unique_ptr<torch::data::StatelessDataLoader<
	    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>,
	    torch::data::samplers::RandomSampler>> test_loader = torch::data::make_data_loader<
	    torch::data::samplers::RandomSampler>(
	    std::move(test_dataset), options.batch_size);

    const unsigned long long test_size = test_dataset.size().value();

    /*Create Feed forward network*/

    const std::shared_ptr<Net> net = std::make_shared<Net>();

    // torch::load(net, "net.pt"); /*To use trained model*/

    /*Using stochastic gradient descent optimizer with learning rate 0.01*/
    torch::optim::SGD optimizer(net->parameters(), 0.01); // Learning Rate 0.01

    for (size_t i = 0; i < options.epochs; i++) 
    {
        /*Run the training for all iterations*/
        train(net, *train_loader, optimizer, i + 1, train_size);
        std::cout << '\n';
        /*Run on the validation set for all iterations*/
        test(net, *test_loader, i + 1, test_size);
        /*Save the network*/
        torch::save(net, "net.pt");
    }

}

void run_inference()
{
	/*Path to Fashion Mnist*/
	const std::string root_string = util::get_dataset_path() + "fashion-mnist/";

	/*Process and load test dat similar to above*/
	torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>> test_dataset =
		CustomDataset(root_string, false).map(torch::data::transforms::Stack<>());

	const std::unique_ptr<torch::data::StatelessDataLoader<
		torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>,
		torch::data::samplers::RandomSampler>> test_loader = torch::data::make_data_loader<
		torch::data::samplers::RandomSampler>(
		std::move(test_dataset), options.batch_size);

	const unsigned long long test_size = test_dataset.size().value();

	std::shared_ptr<Net> net = std::make_shared<Net>();

    //We load the saved network to run inference on test data. Initialize the network and then use
	//torch::load(network, path_to_network);
	torch::load(net, "net.pt");

	test(net, *test_loader, options.epochs, test_size);
}