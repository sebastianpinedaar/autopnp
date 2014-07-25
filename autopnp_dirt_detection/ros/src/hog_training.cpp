/*
 * hog_training.cpp
 *
 *  Created on: Jun 10, 2014
 *      Author: rmb-sp
 *
 *This module processes a image data set determined by "Database.txt". It extracts the HOG Features and makes a data set which is split in Training Data set and test data set, in order to
 *evaluate the performance. Then, this data are used to train a support vector machine (SVM) with linear kernel. Later, the program evaluates the accuracy of the trained machine and
 *outputs this results.

 */


#include <autopnp_dirt_detection/training.h>

int main (){

	cv::HOGDescriptor hog;
	Preprocessing preprocessor;
	TrainingContainer trainer;
	cv::Mat image, im;
	cv::Mat Features;
	cv::Mat circle= cv::Mat::zeros(cv::Size(RES_X,RES_Y),CV_8U);
	std::string name,ans,  letter_pos, letter_neg;
	std::string ROIP, ROIN, name_svm;
	std::vector<cv::Point> locations;
	std::vector<float> featureVector;
	const cv::Size trainingPadding = cv::Size(0,0);
	const cv::Size winStride = cv::Size(8,8);//Strid 16= no overlap
	int cell_size, features_number, len;
	int thickness=-1, lineType=8;
	int p, angle_step;

	//The images name used for training are contained in "Database.txt"
	DataBase db("Database.txt",FALSE);

	std::cout<<"How big should the cell size be (the cell is supposed to be square, enter only the size of a side)?"<<std::endl;
	std::cin>> cell_size;

	std::cout<<"Enter the number of angular positions you wish to make: "<<std::endl;
	std::cin>> p;


	if((360%p)!=0){
		std::cout<<"The number of angular positions must be divisor of 360"<<std::endl;
		exit(0);
	}

	//Sets the Histogram of Oriented Gradients parameters
	hog.blockSize= cv::Size(16,16);
	hog.winSize= cv::Size(RES_X,RES_Y);
	hog.blockStride=cv::Size(8,8);
	hog.cellSize= cv::Size(cell_size, cell_size);

	features_number= 9*(hog.blockSize.width/hog.cellSize.width)*(hog.blockSize.height/hog.cellSize.height)*((hog.winSize.width-hog.blockSize.width)/hog.blockStride.width+1)*((hog.winSize.height-hog.blockSize.height)/hog.blockStride.height+1);
	std::cout<<"Features number:"<<features_number<<std::endl;

	cv::Mat hog_features(db.getSize()*p,features_number+1, CV_32F );
	angle_step= 360/p;

	//Drawing a circle
	cv::circle(circle, cv::Size(circle.cols/2, circle.rows/2),circle.rows/2, 255, thickness, lineType );


	//Information about how the elemnts are labelled
	std::cout<<"Which is the first letter of the name of the negative samples?"<<std::endl;
	std::cin>> letter_neg;

	std::cout<<"Do you want to extract a ROI to the negatives examples?(y/n)"<<std::endl;
	std::cin>>ROIN;


	std::cout<<"Which is the first letter of the name of the positive samples?"<<std::endl;
	std::cin>> letter_pos;

	std::cout<<"Do you want to extract a ROI to the positives examples?(y/n)"<<std::endl;
	std::cin>>ROIP;

	if(db.getSize()==0){
		std::cout<<"The data set doesn't contain a image. Try to fill it with a image name."<<std::endl;
		exit(0);
	}




	for(int i=0; i<db.getSize(); i++) {

		name= db.getElement(i);
		preprocessor.readImage(name+".jpg");

		std::cout<<"Image "<<name<<" loaded..."<<std::endl;

		cv::namedWindow("Image",1);
		cv::imshow("Image",preprocessor.getImage());
		cv::waitKey();


		//Extracts region of interests if it is necessary
		if((ROIN[0]=='y' && name[0]==letter_neg[0]) || (ROIP[0]=='y' && name[0]==letter_pos[0]))
			image=preprocessor.imageExtractROI();
		else
			image=preprocessor.getImage();

		//Resizes in order to get an uniform image
		cv::resize(image, im, cv::Size(RES_X,RES_Y));


		//Mask with a circle to get an uniform image after rotaiotng.
		im.copyTo(image, circle);

		len= std::max(image.cols, image.rows);
		cv::Mat image1(image.rows, image.cols, CV_8UC3);
		cv::Point2f pt(len/2,len/2);



		//Rotating the image p times and getting the features of each one
		for(int n=0; n< p; n++){

			//Determines how much the image will be rotated
			std::cout<<"Getting rotation matrix..."<<std::endl;
			cv::Mat r= cv::getRotationMatrix2D(pt,angle_step*n,1);

			std::cout<<"Rotating..."<<std::endl;

			//Performs the rotation of the image
			cv::warpAffine(image, image1, r, cv::Size(len,len));


			std::cout<<"Changing the size..."<<std::endl;
			std::cout<<"Graphing the rotated image..."<<std::endl;

			//Computes the HOG features
			hog.compute(image1, featureVector, winStride, trainingPadding, locations);

			std::cout<<"Computing features for angle: "<<angle_step*n<<std::endl;

			for(int j=0; j<featureVector.size(); j++){
				hog_features.at<float>(i*p+n,j)= featureVector.at(j);
			}

			//Labels the image
			if(name[0]==letter_neg[0]){
				hog_features.at<float>(i*p+n,featureVector.size())= -1;
			}
			else
				hog_features.at<float>(i*p+n,featureVector.size())= 1;

			std::cout<<"Size of feature vector:"<<featureVector.size()<<std::endl;
			std::cout<<"Computed features..."<<std::endl;

			featureVector.clear();

		}
		image.release();
	}


	std::cout<<"Information about the histogram of oriented gradients extractor"<<std::endl;
	std::cout<<" - Block size: "<<hog.blockSize<<std::endl;
	std::cout<<" - Cell size: "<<hog.cellSize<<std::endl;
	std::cout<<" - Block stride: "<<hog.blockStride<<std::endl;
	std::cout<<"Hog_features size"<<hog_features.size()<<std::endl;


	std::cout<<"Enter the name of the SVM:"<<std::endl;
	std::cin>> name_svm;

	//Performs the trainng and the evaluation
	trainer.setFeatures(hog_features);
	trainer.nameSVM(name_svm);
	trainer.splitting();
	trainer.training();
	trainer.evaluation();

	//Sves the vector machines for further use
	trainer.saveSVM();


	return 0;
}
