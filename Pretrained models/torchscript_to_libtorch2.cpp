#include <stdint.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "readImages.h"

struct Options
{
	int batchSize = 4; //Batch size
	size_t epochs = 5; // Number of epochs
	size_t logInterval = 20;
	std::ofstream loss_acc_train;
	std::ofstream loss_acc_test;
	torch::DeviceType device = torch::kCPU;
};

static Options options;

typedef std::vector<std::string> stringvec;
using Data = std::vector<std::pair<torch::Tensor, torch::Tensor>>;

//ConvertData function from readImages.h, other helper functions can be seen in the code
Data convertData(std::pair<std::vector<stringvec>, std::vector<int>> data_path, bool train)
{
    stringvec::iterator img_name;
    std::vector<int>::iterator l;

    std::vector<stringvec> images_path = data_path.first;
    std::vector<int> labels = data_path.second;
    std::string index;

    Data final_tensorData;

    for (int i = 0; i < images_path.size(); i++) 
    {
        int total_images_count = images_path[i].size();
        int train_images_count = 0.85 * total_images_count;
        int start, end;

        if (train) 
        {
            start = 0;
            end = train_images_count;
            index = "train";
        }
        else 
        {
            start = train_images_count;
            end = total_images_count;
            index = "test";
        }

        for (img_name = images_path[i].begin() + start; img_name != images_path[i].begin() + end; img_name++) 
        {
            cv::Mat image = cv::imread(*img_name);
            // We convert opencv BGR to RGB as pretrained models trained on RGB.
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::Size scale(224, 224);
            //Resize the image to standard size for classification
            cv::resize(image, image, scale, 0, 0, cv::INTER_AREA);

            // Convert Mat image to tensor 1 x H x W x C
            torch::Tensor tensorImage = torch::from_blob(image.data, { image.rows, image.cols, image.channels() }, at::kByte);
            tensorImage = tensorImage.to(torch::kFloat32).div_(255);

            tensorImage = at::transpose(tensorImage, 0, 1);
            tensorImage = at::transpose(tensorImage, 0, 2);

            torch::Tensor tLabel = torch::tensor(i);
            final_tensorData.push_back(std::make_pair(tensorImage, tLabel));
        }
    }

    if (train) 
    {
        std::random_shuffle(final_tensorData.begin(), final_tensorData.end());
    }

    return final_tensorData;
}

Data process_data(std::pair<std::vector<stringvec>, std::vector<int>> root, bool train)
{
	Data tensorData = convertData(root, train);
	return tensorData;
}

//Use CustomDataset class to load any type of dataset other than inbuilt datasets
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
    using Example = torch::data::Example<>;

    Data data;
    size_t data_size;
public:
    CustomDataset(std::pair<std::vector<stringvec>, std::vector<int>> root, bool isTrain)
	{
        data = process_data(root, isTrain);
        data_size = data.size();
        //std::cout << "data" << data.size() << std::endl;
    }

    //Returns the data sample at the given `index
    Example get(size_t index) override
	{
        // This should return {torch::Tensor, torch::Tensor}
        auto img = data[index].first;
        auto label = data[index].second;
        return { img.clone(), label.clone() };
    };

    torch::optional<size_t> size() const override
	{
        return data_size;
    };
};

template <typename DataLoader>
void train(torch::jit::script::Module module, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size)
{
    size_t index = 0;
    //Set network in the training mode
    module.train();
    float Loss = 0, Acc = 0;

    for (torch::data::Example<>& batch : loader)
    {
        const torch::Tensor data = batch.data.to(options.device);
        torch::Tensor targets = batch.target.to(options.device).view({ -1 });

        targets = targets.to(torch::kInt64);

        std::vector<torch::jit::IValue> input;
        input.push_back(data);
        torch::Tensor output = module.forward(input).toTensor();
        output = torch::log_softmax(output, 1);

        //Using negative log likelihood loss function to compute loss
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

    if (index++ % options.logInterval == 0) 
    {
        auto end = data_size;
        options.loss_acc_train << std::to_string(Loss / data_size) + "," + std::to_string(Acc / data_size) << std::endl;

        std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
            << "\tLoss: " << Loss / data_size << "\tAcc: " << Acc / data_size
            << '\n';
    }
}

template <typename DataLoader>
void test(torch::jit::script::Module module, DataLoader& loader, size_t epoch, size_t data_size)
{
    size_t index = 0;
    float Loss = 0, Acc = 0;
    module.eval();

    for (const torch::data::Example<>& batch : loader) 
    {
        torch::Tensor data = batch.data.to(options.device);
        torch::Tensor targets = batch.target.to(options.device).view({ -1 });

        targets = targets.to(torch::kInt64);

        std::vector<torch::jit::IValue> input;
        input.push_back(data);
        torch::Tensor output = module.forward(input).toTensor();
        output = torch::log_softmax(output, 1);

        //Store all prediction failure cases
        if (epoch == options.epochs) 
        {
            for (int j = 0; j < output.size(0); j++) 
            {
                if (targets[j].item<int>() != output[j].argmax().item<int>()) 
                {
                    cv::Mat test_image(224, 224, CV_8UC3);
                    torch::Tensor tensor = data[j].mul_(255).clamp(0, 255).to(torch::kU8);
                    tensor = tensor.to(torch::kCPU);
                    std::memcpy((void*)test_image.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

                    std::cout << "***** TESTING on TEST IMAGE " << j << " *****\n";
                    std::cout << "GroundTruth: " << targets[j].template item<float>()
                        << ", Prediction: " << output[j].argmax() << '\n';
                    std::cout << "Output Probabilities\n";

                    for (int i = 0; i < output[j].size(0); i++) 
                    {
                        std::cout << "Class: " << i << " " << torch::exp(output[j])[i].template item<float>() << '\n';
                    }

                    cv::imwrite("output0/OUTPUT" + std::to_string(j) + "_GT_" + std::to_string(targets[j].template item<int>()) +
                        "_Pred_" + std::to_string(output[j].argmax().template item<int>()) + ".jpg", test_image);
                }
            }

            std::cout << "Outputs saved, Please checkout the output images\n";
        }

        torch::Tensor loss = torch::nll_loss(output, targets);
        torch::Tensor acc = output.argmax(1).eq(targets).sum();

        Loss += loss.item<float>();
        Acc += acc.item<float>();
    }

    if (index++ % options.logInterval == 0) 
    {
        options.loss_acc_test << std::to_string(Loss / data_size) + "," + std::to_string(Acc / data_size) << '\n';
        std::cout << "Val Epoch: " << epoch
            << "\tVal Loss: " << Loss / data_size << "\tVal ACC:" << Acc / data_size << '\n';
    }
}

int convert_main2()
{
    //Use CUDA for computation if available
    std::cout << torch::cuda::is_available() << '\n';
    if (torch::cuda::is_available())
    {
        options.device = torch::kCUDA;
    }
    std::cout << "Running on: " << (options.device == torch::kCUDA ? "CUDA" : "CPU") << '\n';
    //Path to Fashion Mnist
    std::string root_string = "./animals4/";
    bool isTrain = true; //Flag to create train or test data

    // Data data_path = read_images(root_string);
    std::pair<std::vector<stringvec>, std::vector<int>> data_path = read_images(root_string);

    //Uses Custom Dataset Class to load train data. Apply stack collation which takes
    //batch of tensors and stacks them into single tensor along the first dimension
    auto train_dataset = CustomDataset(data_path, isTrain).map(torch::data::transforms::Stack<>());
    //Data Loader provides options to speed up the data loading like batch size, number of workers
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), options.batchSize);
    auto train_size = train_dataset.size().value();

    std::cout << train_size << '\n';

    //Process and load test dat similar to above
    auto test_dataset = CustomDataset(data_path, false).map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset), options.batchSize);
    auto test_size = test_dataset.size().value();

    //Load traced jit module
    torch::jit::script::Module module;
    module = torch::jit::load("vgg16_model0.pt");
    module.to(options.device);

    options.loss_acc_train.open("loss_acc_train0.txt");
    options.loss_acc_test.open("loss_acc_test0.txt");

    std::vector<torch::Tensor> parameters;
    getModuleParams(parameters, module);
    torch::optim::RMSprop optimizer(parameters, 0.00001);

    for (size_t i = 0; i < options.epochs; i++)
    {
        // Run the training for all epochs
        train(module, *train_loader, optimizer, i + 1, train_size);
        std::cout << std::endl;
        //Run on the validation set for all epochs
        test(module, *test_loader, i + 1, test_size);
        //Save the network
        module.save("net0.pt");
    }

    return 0;
}