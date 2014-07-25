/*
 *  hog_recognizer.cpp
 *
 *  Created on: Jun 10, 2014
 *      Author: rmb-sp
 *
 *      This program recognize to kind of objects: pen and scissors from  a background. THe image
 *      is gained directly from the Kinect. Then, the HOG features are extracted from the image and passed to
 *      three binary classifiers which forms a multi-classifier. Each classifiers is a support vector machines whose
 *      parameters are found by training and are loaded from a file.
 */



// ROS includes
#include <ros/ros.h>

// ROS message includes
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>

// topics
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// opencv

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


// point cloud
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>


#include <string>
#include <math.h>

//Flags for resizing
#define RES_X 128
#define RES_Y 128

//Flags for classifcation result
#define NO_IDENTIFIED 0
#define PEN 1
#define SCISSORS 2

//Classifier class contains the methods tu classify the image

class Classifier
{
	cv::Mat color_image;
	cv::Mat extracted_image;
	int c;

public:
	Classifier () {

	}

	void setImage(cv::Mat image){
		color_image=image;
	}

	void imageExtractROI (cv::Mat im) {

		color_image=im;

		int half_side_length = 45;// Half side lenght= 50-(5)

		cv::Mat I (color_image, cv::Rect(cv::Point(color_image.cols/2-half_side_length, color_image.rows/2-half_side_length),
		cv::Point(color_image.cols/2+half_side_length, color_image.rows/2+half_side_length)) ); // using a rectangle

		extracted_image=I;


	}


	void classify(){

		cv::HOGDescriptor hog, hog2;
		cv::Mat image, im;
		std::string name;
		const cv::Size trainingPadding = cv::Size(0,0);
		const cv::Size winStride = cv::Size(8,8);//Strid 16= no overlap
		std::vector<cv::Point> locations;
		std::vector<float> featureVector, featureVector2;
		std::vector<int> voting(3);
		CvSVM SVM_pen, SVM_scissors, SVM_pen_scissors;
		float classification_result1, classification_result2, classification_result3;
		cv::Mat circle= cv::Mat::zeros(cv::Size(RES_X,RES_Y),CV_8U);
		int thickness=-1, lineType=8;
		int max;
		cv::circle(circle, cv::Size(circle.cols/2, circle.rows/2),circle.rows/2, 255, thickness, lineType );

		//filling vector for voting
		std::fill( voting.begin(), voting.end(), 0 );

		hog.blockSize= cv::Size(16,16);
		hog.winSize= cv::Size(RES_X,RES_Y);
		hog.blockStride=cv::Size(8,8);
		hog.cellSize= cv::Size(8,8);

		hog2.blockSize= cv::Size(16,16);
		hog2.winSize= cv::Size(RES_X,RES_Y);
		hog2.blockStride=cv::Size(8,8);
		hog2.cellSize= cv::Size(16,16);

		//Loading support vector machines
		SVM_scissors.load("svm_scissors");
		SVM_pen.load("svm_pen");
		SVM_pen_scissors.load("pen_scissors_svm");


		image=extracted_image;

		cv::resize(image, im, cv::Size(RES_X,RES_Y));
		im.copyTo(image, circle);


		//Finding features
		hog.compute(image, featureVector, winStride, trainingPadding, locations);
		hog2.compute(image, featureVector2, winStride, trainingPadding, locations);

		cv::Mat features(1, featureVector.size(), CV_32F);
		cv::Mat features2(1, featureVector2.size(), CV_32F);


		for(int j=0; j<featureVector.size(); j++){
			features.at<float>(0,j)= featureVector.at(j);
		}

		for(int j=0; j<featureVector2.size(); j++){
			features2.at<float>(0,j)= featureVector2.at(j);
		}

		//Evaluating features in the support vector machine
		classification_result1=  SVM_scissors.predict(features);
		classification_result2= SVM_pen.predict(features);
		classification_result3= SVM_pen_scissors.predict(features2);

		std::cout<<"Output of scissors classifier:"<<classification_result1<<std::endl;
		std::cout<<"Output of pen classifier:"<<classification_result2<<std::endl;
		std::cout<<"Output of scissors/pen classifier:"<<classification_result3<<std::endl;


		//Deciding the classification result

		classification_result3= SVM_pen_scissors.predict(features2);

		if( classification_result3==1){
			classification_result1=  SVM_scissors.predict(features);
			if(classification_result1==1) {
				std::cout<<"IT IS A SCISSORS."<<std::endl;
				c= SCISSORS;
			}
			else{
				std::cout<<"IT IS NEITHER A SCISSOPRS NOR A PEN"<<std::endl;
				c= NO_IDENTIFIED;
			}

		}
		else {
			classification_result2= SVM_pen.predict(features);
			if(classification_result2==1) {
				std::cout<<"IT IS A PEN."<<std::endl;
				c= PEN;
			}
			else{
				std::cout<<"IT IS NEITHER A SCISSORS NOR A PEN."<<std::endl;
				c= NO_IDENTIFIED;
			}
		}
	}


	int getClass() {
		return c;
	}

	~Classifier(){
	//	std::cout<<"Detroying preprocessing.."<<std::endl;
	}
};


class RGBDRecording
{
public:
	RGBDRecording(ros::NodeHandle nh) :
			node_handle_(nh)
	{
		it_ = 0;
		sync_input_ = 0;

		it_ = new image_transport::ImageTransport(node_handle_);
		colorimage_sub_.subscribe(*it_, "colorimage_in", 1);
		pointcloud_sub_.subscribe(node_handle_, "pointcloud_in", 1);

		sync_input_ = new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> >(30);
		sync_input_->connectInput(colorimage_sub_, pointcloud_sub_);
		sync_input_->registerCallback(boost::bind(&RGBDRecording::inputCallback, this, _1, _2));
	}

	~RGBDRecording()
	{
		if (it_ != 0)
			delete it_;
		if (sync_input_ != 0)
			delete sync_input_;
	}

// Converts a color image message to cv::Mat format.
	void
	convertColorImageMessageToMat(const sensor_msgs::Image::ConstPtr& image_msg, cv_bridge::CvImageConstPtr& image_ptr, cv::Mat& image)
	{
		try
		{
			image_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("PeopleDetection: cv_bridge exception: %s", e.what());
		}
		image = image_ptr->image;
	}

	void
	inputCallback(const sensor_msgs::Image::ConstPtr& color_image_msg, const sensor_msgs::PointCloud2::ConstPtr& pointcloud_msg)
	{
		//ROS_INFO("Input Callback");

		//Elements needed for recognition




		// convert color image to cv::Mat
		cv_bridge::CvImageConstPtr color_image_ptr;
		cv::Mat color_image;
		convertColorImageMessageToMat(color_image_msg, color_image_ptr, color_image);

		// get color image from point cloud
		pcl::PointCloud<pcl::PointXYZRGB> point_cloud_src;
		pcl::fromROSMsg(*pointcloud_msg, point_cloud_src);

// cv::Mat color_image = cv::Mat::zeros(point_cloud_src.height, point_cloud_src.width, CV_8UC3);
// for (unsigned int v=0; v<point_cloud_src.height; v++)
// {
// for (unsigned int u=0; u<point_cloud_src.width; u++)
// {
// pcl::PointXYZRGB point = point_cloud_src(u,v);
// if (isnan_(point.z) == false)
// color_image.at<cv::Point3_<unsigned char> >(v,u) = cv::Point3_<unsigned char>(point.b, point.g, point.r);
// }
// }
		cv::Mat color_image_copy = color_image.clone();
		int half_side_length = 50;
		cv::rectangle(color_image_copy, cv::Point(color_image.cols/2-half_side_length, color_image.rows/2-half_side_length),
				cv::Point(color_image.cols/2+half_side_length, color_image.rows/2+half_side_length), CV_RGB(0, 255, 0), 3);
		cv::imshow("color image", color_image_copy);

		char key= cv::waitKey(20);
		//std::cout<<key;
		//char key = 'c';
		//std::cout<<"gut"<<std::endl;

		//std::cout<< "How do you want to name the image? (The name of the data cloud is the same as the one of the image"<<std::endl;


		if (key=='c')
		{
			//The image captured is in 'color_image_copy' and then classified
			Classifier C;
			C.imageExtractROI(color_image_copy);
			C.classify();

		}
		else if (key=='q')
			ros::shutdown();


	}

private:
	ros::NodeHandle node_handle_;

	// messages
	image_transport::ImageTransport* it_;
	image_transport::SubscriberFilter colorimage_sub_; ///< Color camera image topic
	message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub_;
	message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> >* sync_input_;
};

int
main(int argc, char** argv)
{
// Initialize ROS, specify name of node
	ros::init(argc, argv, "cob_surface_classification");

// Create a handle for this node, initialize node
	ros::NodeHandle nh;

// Create and initialize an instance of CameraDriver
	RGBDRecording surfaceClassification(nh);

	ros::spin();

	return (0);
}
