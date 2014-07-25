// ROS includes
#include <ros/ros.h>
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/legacy/legacy.hpp>
#include <sensor_msgs/image_encodings.h>



#include <string>
#include <math.h>
#include <vector>

#define RES_X 128
#define RES_Y 128

#define NO_IDENTIFIED 0
#define PEN 1
#define SCISSORS 2

#define TRAIN_RATIO 0.8


//Class to deal with list of images saved in a file
class DataBase {

	std::string file_name;
	std::fstream file;
	std::vector<std::string> elements;
	bool new_file;

public:


/*	The database is a list of image saved in a file. The constuctor receives the name of the file containing the elements which forms the database.
 *  The boolean variable new_file_ determines whether this file could be writen (1) or not (0).
 *  */

	DataBase(std::string file_name_, bool new_file_);

	//the number of images contained in the database
	int getSize() ;

	//add another element to the list
	void addElement (std::string element);

	std::string getElement(int n) ;

	//loads a new database (a new file)
	void setDataBaseName (std::string file_name_);


	std::string getDataBaseName () ;
	~DataBase();
};



//Class to classify images
class Classifier
{
	cv::Mat color_image;
	cv::Mat extracted_image;
	int c;

public:
	Classifier ();
	//Set the image to process, use it if you dont want to extract a ROI
	void setImage(cv::Mat image);

	//Extract the region of interest and set the image
	void imageExtractROI (cv::Mat im) ;

	//Classify the image
	void classify();

	//Get the class of the classified iage
	int getClass() ;

	~Classifier();
};

//Class to train classifier
class TrainingContainer {

	cv::string name_SVM;
	CvSVM SVM;
	cv::Mat features;
	cv::Mat training_data;
	cv::Mat test_data;
	cv::Mat training_labels;
	cv::Mat test_labels;


public:

	TrainingContainer () ;
	TrainingContainer (cv::Mat td, cv::Mat testd, cv::Mat tl, cv::Mat testl);


	void setData(cv::Mat td, cv::Mat testd, cv::Mat tl, cv::Mat testl);

	//Set of features adquired by HOG
	void setFeatures(cv::Mat f);

	//Name of the file which contains the information of the learner
	void nameSVM(std::string n) ;

	cv::Mat getTrainingData();
	cv::Mat getTestData();
	cv::Mat getTrainingLabels();
	cv::Mat getTestLabels();

	//"Random vector" a function used to sort randomly a set of integer number. The integer number to be sorted are from 0 to m. It returns the vector with this numbers randomly sorted.
	std::vector <int> random_vector (int m) ;


	//Splits the data set in false positives and negatives
	void splitting();

	//Save the SVM informaiton
	void saveSVM();

	//Evaluates the performance of the learner on the training set and test set
	void evaluation() ;

	//Train the SVM according to the features extracted and the labels.
	void training ();



};

//Class to process images
class Preprocessing
{
	cv::Mat color_image;
	cv::Mat extracted_image;
	std::string image_name;

public:
	Preprocessing () ;

	void readImage (std::string image_name_) ;

	cv::Mat imageExtractROI () ;

	cv::Mat getImage();
};
