#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <memory>
#include <utility>

#include "CSVReader.h"
#include "env_util.h"

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

//Linear Regression Example
struct Options
{
    size_t train_batch_size = 4;
    size_t test_batch_size = 100;
    size_t epochs = 1000;
    size_t log_interval = 20;
    // path must end in delimiter
    std::string dataset_path = util::get_dataset_path() + "/BostonHousing.csv";
    // For CPU use torch::kCPU and for GPU use torch::kCUDA
    torch::DeviceType device = torch::kCUDA;
};

static Options options;

std::vector<std::vector<float>> normalize_feature(std::vector<std::vector<std::string> > feat, int rows, int cols) {
	const std::vector<float> input(cols, 1);
    std::vector<std::vector<float>> data(rows, input);

    for (int i = 0; i < cols; i++) 
    {   // each column has one feature
        // initialize the maximum element with 0 
        // std::stof is used to convert string to float
        float maxm = std::stof(feat[1][i]);
        float minm = std::stof(feat[1][i]);

        // Run the inner loop over rows (all values of the feature) for given column (feature) 
        for (int j = 1; j < rows; j++) 
        {
            // check if any element is greater  
            // than the maximum element 
            // of the column and replace it 
            if (std::stof(feat[j][i]) > maxm)
            {
                maxm = std::stof(feat[j][i]);
            }

            if (std::stof(feat[j][i]) < minm)
            {
                minm = std::stof(feat[j][i]);
            }
        }

        // From above loop, we have min and max value of the feature
        // Will use min and max value to normalize values of the feature
        for (int j = 0; j < rows - 1; j++) 
        {
            // Normalize the feature values to lie between 0 and 1
            data[j][i] = (std::stof(feat[j + 1][i]) - minm) / (maxm - minm);
        }
    }

    return data;
}

// Define Data to accomodate pairs of (input_features, output)
using Data = std::vector<std::pair<std::vector<float>, float>>;

class CustomDataset final : public torch::data::datasets::Dataset<CustomDataset>
{
    using Example = torch::data::Example<>;

    Data data_;

public:
    CustomDataset(const Data& data) : data_(data) {}

    // Returns the Example at the given index, here we convert our data to tensors
    Example get(const size_t index) override
    {
        int f_size = data_[index].first.size();
        // Convert feature vector into tensor of size fSize x 1
        torch::Tensor tdata = torch::from_blob(&data_[index].first, {f_size, 1});
        // Convert output value into tensor of size 1
        torch::Tensor toutput = torch::from_blob(&data_[index].second, {1});
        return { tdata, toutput };
    }

    // To get the size of the data
    torch::optional<size_t> size() const override
    {
        return data_.size();
    }
};

std::pair<Data, Data> read_info()
{
	Data train;
	Data test;

	// Reads data from CSV file.
    // CSVReader class is defined in CSVReader.h header file
    CSVReader reader(options.dataset_path);
	const std::vector<std::vector<std::string>> data_list = reader.getData();

	const int n = data_list.size(); // Total number of data points
    // As last column is the output, feature size will be number of column minus one.
	const int f_size = data_list[0].size() - 1;
    std::cout << "Total number of features: " << f_size << '\n';
    std::cout << "Total number of data points: " << n << '\n';
	const int limit = 0.8 * n;    // 80 percent data for training and rest 20 percent for validation
    std::vector<float> input(f_size, 1);
    std::vector<std::vector<float>> data(n, input);

    // Normalize data
    data = normalize_feature(data_list, n, f_size);

    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < f_size; j++) 
        {
            input[j] = data[i - 1][j];
        }

        float output = std::stof(data_list[i][f_size]);

        // Split data data into train and test set
        if (i <= limit)
        {
            train.push_back({ input, output });
        }
        else 
        {
            test.push_back({ input, output });
        }
    }

    std::cout << "Total number of training data: " << train.size() << std::endl;
    std::cout << "Total number of test data: " << test.size() << std::endl;

    // Shuffle training data
    std::random_device rd;
    std::mt19937 seed(rd());

    std::shuffle(train.begin(), train.end(), seed);

    return std::make_pair(train, test);
}

// Linear Regression Model
struct Net final : torch::nn::Module
{
    /*
    Network for Linear Regression is just a single neuron (i.e. one Dense Layer)
    Usage: auto net = std::make_shared<Net>(num_features, num_outputs)
    */
    Net(int num_features, int num_outputs)
	{
        neuron = register_module("neuron", torch::nn::Linear(num_features, num_outputs));
    }

    torch::Tensor forward(torch::Tensor x)
	{
        /*Convert row tensor to column tensor*/
        x = x.reshape({ x.size(0), -1 });
        /*Pass the input tensor through linear function*/
        x = neuron->forward(x);
        return x;
    }

    /*Initilaize the constructor with null pointer. More details given in the reference*/
    torch::nn::Linear neuron{ nullptr };
};

template <typename DataLoader>
void train(const std::shared_ptr<Net>& network, DataLoader& loader, torch::optim::Optimizer& optimizer,
const size_t epoch, const size_t data_size)
{
    size_t index = 0;
    /*Set network in the training mode*/
    network->train();
    float Loss = 0;

    for (auto& batch : loader)
    {
	    const torch::Tensor data = batch.data.to(options.device);
        torch::Tensor targets = batch.target.to(options.device).view({ -1 });
        // Execute the model on the input data

        torch::Tensor output = network->forward(data);

        //Using mean square error loss function to compute loss
        torch::Tensor loss = torch::mse_loss(output, targets);

        // Reset gradients
        optimizer.zero_grad();
        // Compute gradients
        loss.backward();
        //Update the parameters
        optimizer.step();

        Loss += loss.item<float>();

        if (index++ % options.log_interval == 0) 
        {
	        const unsigned long long end = std::min(data_size, (index + 1) * options.train_batch_size);

            std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                << "\t\tLoss: " << Loss / end << '\n';
        }
    }
}

template <typename DataLoader>
void test(const std::shared_ptr<Net>& network, DataLoader& loader, size_t data_size)
{
    network->eval();

    for (const auto& batch : loader)
    {
	    const torch::Tensor data = batch.data.to(options.device);
        const torch::Tensor targets = batch.target.to(options.device).view({ -1 });

        const torch::Tensor output = network->forward(data);
        std::cout << "Predicted:" << output[0].item<float>() << "\t" << "Ground truth: "
            << targets[1].item<float>() << '\n';
        std::cout << "Predicted:" << output[1].item<float>() << "\t" << "Ground truth: "
            << targets[1].item<float>() << '\n';
        std::cout << "Predicted:" << output[2].item<float>() << "\t" << "Ground truth: "
            << targets[2].item<float>() << '\n';
        std::cout << "Predicted:" << output[3].item<float>() << "\t" << "Ground truth: "
            << targets[3].item<float>() << '\n';
        std::cout << "Predicted:" << output[4].item<float>() << "\t" << "Ground truth: "
            << targets[4].item<float>() << '\n';

        torch::Tensor loss = torch::mse_loss(output, targets);

        break;
    }
}

void linear_regression()
{
    /*Sets manual seed from libtorch random number generators*/
    torch::manual_seed(1);

    /*Use CUDA for computation if available*/
    if (torch::cuda::is_available())
    {
        options.device = torch::kCUDA;
    }

    std::cout << "Running on: " << (options.device == torch::kCUDA ? "CUDA" : "CPU") << '\n';

    /*Read data and split data into train and test sets*/
    const std::pair<Data, Data> data = read_info();

    /*Uses Custom Dataset Class to load train data. Apply stack collation which takes
    batch of tensors and stacks them into single tensor along the first dimension*/
    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>> train_set =
	    CustomDataset(data.first).map(torch::data::transforms::Stack<>());
    const unsigned long long train_size = train_set.size().value();

    /*Data Loader provides options to speed up the data loading like batch size, number of workers*/
    const std::unique_ptr<torch::data::StatelessDataLoader<
	    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>,
	    torch::data::samplers::RandomSampler>> train_loader = torch::data::make_data_loader(
	    std::move(train_set), options.train_batch_size);

    std::cout << train_size << '\n';
    /*Uses Custom Dataset Class to load test data. Apply stack collation which takes
    batch of tensors and stacks them into single tensor along the first dimension*/
    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>> test_set =
	    CustomDataset(data.second).map(torch::data::transforms::Stack<>());
    const unsigned long long test_size = test_set.size().value();

    /*Test data loader similar to train data loader*/
    const std::unique_ptr<torch::data::StatelessDataLoader<
	    torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>,
	    torch::data::samplers::RandomSampler>> test_loader =
	    torch::data::make_data_loader(
		    std::move(test_set), options.test_batch_size);

    /*Create Linear  Regression Network*/
    const std::shared_ptr<Net> net = std::make_shared<Net>(13, 1);

    /*Moving model parameters to correct device*/
    net->to(options.device);
    /*Using stochastic gradient descent optimizer with learning rate 0.000001*/
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.000001));

    std::cout << "Training...\n";
    for (size_t i = 0; i < options.epochs; ++i) 
    {
        /*Run the training for all iterations*/
        train(net, *train_loader, optimizer, i + 1, train_size);
        std::cout << '\n';

        if (i == options.epochs - 1) 
        {
            std::cout << "Testing...\n";
            test(net, *test_loader, test_size);
        }
    }
}