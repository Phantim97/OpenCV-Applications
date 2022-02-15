#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int tscript_to_ltorch()
{
    if (argc != 2) 
    {
        std::cerr << "Usage: ./pretrained_models <path-to-exported-script-module> <path-to-image-file (default: dog.jpg)>\n";
        return -1;
    }
    std::string image_file = "dog.jpg";
    if (argc == 3)
    {
        image_file = argv[2];
    }

    // filestream variable file 
    std::ifstream input("imagenet_classes.txt");
    std::string line;
    std::list<std::string> classes;

    // extracting classes from the file 
    while (std::getline(input, line))
    {
        classes.push_back(line);
    }
    //https://pytorch.org/docs/master/jit.html
    //for creating models in python and use in C++ indepedently
    torch::jit::script::Module module;

    try 
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    //Set the model in evaluation mode
    module.eval();

    //Load the image locally
    cv::Mat testImage = cv::imread(image_file);
    cv::Size scale(224, 224);
    //Resize the image to standard size for classification
    cv::resize(testImage, testImage, scale, 0, 0, cv::INTER_AREA);

    // Check if image is not empty
    if (!testImage.empty()) 
    {
        // Convert Mat image to tensor 1 x C x H x W 
        at::Tensor imageTensor = imageToTensor(testImage);

        //Initialize normalize function with imagenet mean and stdev values
        Normalize normalizeChannels({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 });
        //Normalize the tensor image
        imageTensor = normalizeChannels(imageTensor);

        // Create a vector of inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(imageTensor);

        // Execute the model and turn its output into a tensor.
        auto output = module.forward(inputs).toTensor();
        //Apply softmax function to get probabilities of output classes
        output = torch::softmax(output, 1);

        auto topClass = torch::max(output, 1);

        auto class_index = std::get<1>(topClass);
        auto score = std::get<0>(topClass);

        auto classname = std::next(classes.begin(), class_index.item<int>());

        //Predicted class name and confidence score
        std::cout << "Class: " << *classname << std::endl;
        std::cout << "Confidence Score: " << score.item<float>() << std::endl;
    }
}