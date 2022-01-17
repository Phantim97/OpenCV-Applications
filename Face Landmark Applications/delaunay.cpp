#include <fstream>
#include <opencv2/opencv.hpp>
#include "env_util.h"

// In a vector of points, find the index of point closest to input point.
static int find_index(const std::vector<cv::Point2f>& points, const cv::Point2f& point)
{
    int min_index = 0;
    double min_distance = cv::norm(points[0] - point);

    for (int i = 1; i < points.size(); i++)
    {
	    const double distance = cv::norm(points[i] - point);

    	if (distance < min_distance)
        {
            min_index = i;
            min_distance = distance;
        }
    }

    return min_index;
}

// Write delaunay triangles to file
static void write_delaunay(const cv::Subdiv2D& subdiv, const std::vector<cv::Point2f>& points, const std::string& filename)
{

    // Open file for writing
    std::ofstream ofs;
    ofs.open(filename);

    // Obtain the list of triangles.
    // Each triangle is stored as vector of 6 coordinates
    // (x0, y0, x1, y1, x2, y2)
    std::vector<cv::Vec6f> triangle_list;
    subdiv.getTriangleList(triangle_list);

    // Will convert triangle representation to three vertices
    std::vector<cv::Point2f> vertices(3);

    // Loop over all triangles
    for (size_t i = 0; i < triangle_list.size(); i++)
    {
        // Obtain current triangle
        cv::Vec6f t = triangle_list[i];

        // Extract vertices of current triangle
        vertices[0] = cv::Point2f(t[0], t[1]);
        vertices[1] = cv::Point2f(t[2], t[3]);
        vertices[2] = cv::Point2f(t[4], t[5]);

        // Find landmark indices of vertices in the points list
        const int landmark1 = find_index(points, vertices[0]);
        const int landmark2 = find_index(points, vertices[1]);
        const int landmark3 = find_index(points, vertices[2]);
        // save to file.

        ofs << landmark1 << " " << landmark2 << " " << landmark3 << '\n';

    }
    ofs.close();
}

static void draw_point(cv::Mat& img, const cv::Point2f fp, const cv::Scalar color)
{
    cv::circle(img, fp, 2, color, cv::FILLED, cv::LINE_AA, 0);
}

// Draw delaunay triangles
static void draw_delaunay(cv::Mat& img, const cv::Subdiv2D& subdiv, const cv::Scalar delaunay_color)
{
    // Obtain the list of triangles.
    // Each triangle is stored as vector of 6 coordinates
    // (x0, y0, x1, y1, x2, y2)
    std::vector<cv::Vec6f> triangle_list;
    subdiv.getTriangleList(triangle_list);

    // Will convert triangle representation to three vertices
    std::vector<cv::Point> vertices(3);

    // Get size of the image
    const cv::Size size = img.size();
    const cv::Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangle_list.size(); i++)
    {
        // Get current triangle
        cv::Vec6f t = triangle_list[i];

        // Convert triangle to vertices
        vertices[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
        vertices[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
        vertices[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

        // Draw triangles that are completely inside the image.
        if (rect.contains(vertices[0]) && rect.contains(vertices[1]) && rect.contains(vertices[2]))
        {
            cv::line(img, vertices[0], vertices[1], delaunay_color, 1, cv::LINE_AA, 0);
            cv::line(img, vertices[1], vertices[2], delaunay_color, 1, cv::LINE_AA, 0);
            cv::line(img, vertices[2], vertices[0], delaunay_color, 1, cv::LINE_AA, 0);
        }
    }
}

//Draw voronoi diagrams
static void draw_voronoi(cv::Mat& img, cv::Subdiv2D& subdiv)
{
    // Vector of voronoi facets.
    std::vector<std::vector<cv::Point2f> > facets;

    // Voronoi centers
    std::vector<cv::Point2f> centers;

    // Get facets and centers
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

    // Variable for the ith facet used by fillConvexPoly
    std::vector<cv::Point> ifacet;

    // Variable for the ith facet used by polylines.
    std::vector<std::vector<cv::Point>> ifacets(1);

    for (size_t i = 0; i < facets.size(); i++)
    {
        // Extract ith facet
        ifacet.resize(facets[i].size());

        for (size_t j = 0; j < facets[i].size(); j++)
        {
            ifacet[j] = facets[i][j];
        }

        // Generate random color
        cv::Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;

        // Fill facet with a random color
        cv::fillConvexPoly(img, ifacet, color, 8, 0);

        // Draw facet boundary
        ifacets[0] = ifacet;
        cv::polylines(img, ifacets, true, cv::Scalar(), 1, cv::LINE_AA, 0);

        // Draw centers.
        cv::circle(img, centers[i], 3, cv::Scalar(), cv::FILLED, cv::LINE_AA, 0);
    }
}

void calculate_delaunay_triangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunay_tri)
{
    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }

    // Get Delaunay triangulation
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    // Variable to store a triangle ( 3 points )
    std::vector<cv::Point2f> pt(3);

    // Variable to store a triangle as indices from list of points
    std::vector<int> ind(3);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        // The triangle returned by getTriangleList is
        // a list of 6 coordinates of the 3 points in
        // x1, y1, x2, y2, x3, y3 format.
        cv::Vec6f t = triangleList[i];

        // Store triangle as a vector of three points
        pt[0] = cv::Point2f(t[0], t[1]);
        pt[1] = cv::Point2f(t[2], t[3]);
        pt[2] = cv::Point2f(t[4], t[5]);


        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            // Find the index of each vertex in the points list
            for (int j = 0; j < 3; j++)
            {
                ind[j] = find_index(points, pt[j]);
            }
            // Store triangulation as a list of indices
            delaunay_tri.push_back(ind);
        }
    }
}

void delaunay_main()
{
    // Create a vector of points.
    std::vector<cv::Point2f> points;

    // Read in the points from a text file
    std::string points_filename(util::get_data_path() + "images/smiling-man-delaunay.txt");
    std::ifstream ifs(points_filename);
    int x, y;
    while (ifs >> x >> y)
    {
        points.emplace_back(x, y);
    }

    std::cout << "Reading file " << points_filename << '\n';

    // Find bounding box enclosing the points.
    cv::Rect rect = cv::boundingRect(points);

    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }

    // Output filename
    std::string output_file_name("results/smiling-man-delaunay.tri");

    // Write delaunay triangles
    write_delaunay(subdiv, points, output_file_name);

    std::cout << "Writing Delaunay triangles to " << output_file_name << '\n';

    // Define colors for drawing.
    cv::Scalar delaunay_color(255, 255, 255), pointsColor(0, 0, 255);
    // Read in the image.
    cv::Mat img = cv::imread(util::get_data_path() + "images/smiling-man.jpg");
    // Rectangle to be used with Subdiv2D
    cv::Size size = img.size();
    cv::Rect rect2(0, 0, size.width, size.height);

    // Create an instance of Subdiv2D
    cv::Subdiv2D subdiv2(rect2);

    // Create a vector of points.
    std::vector<cv::Point2f> points2;

    // Read in the points from a text file
    std::ifstream ifs2(util::get_data_path() + "images/smiling-man-delaunay.txt");

	int x2;
    int y2;

    while (ifs2 >> x2 >> y2)
    {
        points2.emplace_back(x2, y2);
    }

    // Image for displaying Delaunay Triangulation
    cv::Mat img_delaunay;

    // Image for displaying Voronoi Diagram.
    cv::Mat img_voronoi = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

    // Final side-by-side display.
    cv::Mat img_display;

    // Insert points into subdiv and animate
    for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);

        img_delaunay = img.clone();
        img_voronoi = cv::Scalar(0, 0, 0);

        // Draw delaunay triangles
        draw_delaunay(img_delaunay, subdiv, delaunay_color);

        // Draw points
        for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
        {
            draw_point(img_delaunay, *it, pointsColor);
        }

        // Draw voronoi map
        draw_voronoi(img_voronoi, subdiv2);

        cv::hconcat(img_delaunay, img_voronoi, img_display);
        cv::imshow("window", img_display);
        cv::waitKey(250);
    }

    // Write delaunay triangles
    write_delaunay(subdiv, points, "smiling-man-delaunay.tri");
}