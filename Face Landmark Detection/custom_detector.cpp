#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include "env_util.h"

double interocular_distance(const dlib::full_object_detection& det)
{
    dlib::vector<double, 2> l, r;
    double cnt = 0;
    // Find the center of the left eye by averaging the points around 
    // the eye.
    for (unsigned long i = 36; i <= 41; ++i)
    {
        l += det.part(i);
        ++cnt;
    }

    l /= cnt;

    // Find the center of the right eye by averaging the points around 
    // the eye.
    cnt = 0;

    for (unsigned long i = 42; i <= 47; ++i)
    {
        r += det.part(i);
        ++cnt;
    }

    r /= cnt;

    // Now return the distance between the centers of the eyes
    return length(l - r);
}

std::vector<std::vector<double> > get_interocular_distances(const std::vector<std::vector<dlib::full_object_detection> >& objects)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

void custom_detector_main()
{
    std::cout << "Select point model (1: 33 points || 2: 70 points): ";
    int opt = 0;
    while (opt != 1 && opt != 2)
    {
        std::cin >> opt;
        if (std::cin.fail())
        {
            std::cout << "Invalid input\n";
            std::cin.clear();
            std::cin.ignore(999999, '\n');
        }

        if (opt != 1 && opt != 2)
        {
            std::cout << "Invalid option please enter 1 or 2\n";
        }
    }

    const std::string point_models[] = { "33", "70" };

	const std::string face_landmark_data_dir = util::get_dataset_path() + "facial_landmark_data";
	const std::string num_points = point_models[opt-1];
	const std::string model_name = "shape_predictor_" + num_points + "_face_landmarks.dat";
    const std::string model_path = face_landmark_data_dir + "/" + model_name;

    dlib::shape_predictor_trainer trainer;
    trainer.set_num_threads(8);
    trainer.set_cascade_depth(10);
    trainer.set_tree_depth(4);
    trainer.set_nu(0.1);
    trainer.set_oversampling_amount(20);
    trainer.set_feature_pool_size(400);
    trainer.set_feature_pool_region_padding(0);
    trainer.set_lambda(0.1);
    trainer.set_num_test_splits(20);

    // Tell the trainer to print status messages to the console so we can
    // see training options and how long the training will take.
    trainer.be_verbose();

    // Now we will create the variables that will hold our dataset.
    // images_train will hold training images and faces_train holds
    // the locations and poses of each face in the training images.
    dlib::array<dlib::array2d<unsigned char>> images_train;
    dlib::array<dlib::array2d<unsigned char>> images_test;

    std::vector<std::vector<dlib::full_object_detection>> faces_train;
    std::vector<std::vector<dlib::full_object_detection>> faces_test;

    // Now we load the data.  These XML files list the images in each
    // dataset and also contain the positions of the face boxes and
    // landmarks (called parts in the XML file).
    load_image_dataset(images_train, faces_train, face_landmark_data_dir + "/training_with_face_landmarks.xml");
    load_image_dataset(images_test, faces_test, face_landmark_data_dir + "/testing_with_face_landmarks.xml");

    // Now finally generate the shape model
    dlib::shape_predictor sp = trainer.train(images_train, faces_train);

    // Now that we have a model we can test it. This function measures the
    // average distance between a face landmark output by the
    // shape_predictor and ground truth data.
    // Note that there is an optional 4th argument that lets us normalize the
    // distances.  Here we are normalizing the error using the interocular
    // distance, as is customary when evaluating face landmarking systems.
    std::cout << "mean training error: " << test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << '\n';

    // The real test is to see how well it does on data it wasn't trained
    // on.
    std::cout << "mean testing error:  " << test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << '\n';

    // Finally, we save the model to disk so we can use it later.
    dlib::serialize(model_path) << sp;
}