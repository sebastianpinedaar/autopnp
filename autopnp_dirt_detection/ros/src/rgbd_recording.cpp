/*!
 *****************************************************************
 * \file
 *
 * \note
 * Copyright (c) 2013 \n
 * Fraunhofer Institute for Manufacturing Engineering
 * and Automation (IPA) \n\n
 *
 *****************************************************************
 *
 * \note
 * Project name: Care-O-bot
 * \note
 * ROS stack name: cob_object_perception
 * \note
 * ROS package name: cob_surface_classification
 *
 * \author
 * Author: Richard Bormann
 * \author
 * Supervised by:
 *
 * \date Date of creation: 07.08.2012
 *
 * \brief
 * functions for display of people detections
 *
 *****************************************************************
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer. \n
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. \n
 * - Neither the name of the Fraunhofer Institute for Manufacturing
 * Engineering and Automation (IPA) nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission. \n
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License LGPL as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License LGPL for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License LGPL along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

// ROS includes
#include <ros/ros.h>

// ROS message includes
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

// topics
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// opencv
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// boost
#include <boost/bind.hpp>

// point cloud
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

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
			std::string path ( "Testing/");
			std::string name;
			std::string save_image;

			std::cout<<std::endl<< "Do you want to save the image? (Y/N)"<<std::endl;
			std::cin>> save_image;			
			
			if(save_image=="Y" || save_image=="YES" || save_image=="y" || save_image=="yes") {
				std::cout<<std::endl<< "How do you want to name the image? (The name of the data cloud is the same as the one of the image.)"<<std::endl;
				std::cin>> name;
				cv::Mat ref=cv::imread(path+name+".jpg");
				
				while (ref.data!=NULL) {
					ROS_INFO("Image name already exist. Please choose another one.");	
					std::cin>> name;
					ref=cv::imread(path+name+".jpg");	

				}			


			///Make sure that image name doesn't exist before
			///	if(ref.data==NULL)
			///	{
			
			///save image on disc
				try {
					imwrite(path+name+".jpg", color_image_copy );
					ROS_INFO("Image captured");
				}
				catch (std::runtime_error& ex) {
					fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
					
				}

			///save point clud on disk
				pcl::io::savePCDFileASCII(path+name+".pcd", point_cloud_src);

				std::cerr << "Saved" << point_cloud_src.points.size () << "data points" << std::endl;
			///	}
			///	else
			///		ROS_INFO("Image name already exist. Please choose another one.");

			// todo: save point cloud and image to disc
			// Bilder: cv::imwrite()
			// Point cloud:  pcd writer oder so ähnlich   pcl::io::savePCDFile()
			// path = "~/.ros/rgbd_recording/"
				}
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
