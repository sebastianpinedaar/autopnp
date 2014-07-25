/*
 * This file contains the mehotds definitions of the classes contained in the file training.h T
 * The functions and methods that are defined here are used by hog_training.cpp
 */

#include <autopnp_dirt_detection/training.h>


	/*The constuctor receives the name of the file containing the elements which forms the database. The boolean variable new_file_ determins whether this file could be writen (1) or not (0)*/
	DataBase::DataBase(std::string file_name_, bool new_file_): file_name(file_name_),
												  new_file(new_file_){

		std::cout<<"Creating the database..."<<std::endl;

		std::string line;

		if(!new_file)
			file.open(file_name_.c_str(), std::fstream::in | std::fstream::out );
		else
			file.open(file_name_.c_str(), std::ofstream::out | std::ofstream::trunc);

		while(getline(file, line)){
			elements.push_back(line);
			}

		file.clear();

		std::cout<<"Database "<<file_name_<<"created ..."<<std::endl;
	}

	int DataBase::getSize() {
		return elements.size();
	}

	void DataBase::addElement (std::string element){


		file<<element.c_str()<<std::endl;
		elements.push_back(element);
		file.clear();

		std::cout<<"Element added successfully..."<<std::endl;
	}

	std::string DataBase::getElement(int n) {
		return elements[n];
	}

	void DataBase::setDataBaseName (std::string file_name_){


		std::string line;

		file.close();
		elements.clear();
		file_name= file_name_;

		if(!new_file)
			file.open(file_name_.c_str(), std::fstream::in | std::fstream::out );
		else
			file.open(file_name_.c_str(), std::ofstream::out | std::ofstream::trunc);

		while(getline(file, line)){
			elements.push_back(line);
		}

		file.clear();

		std::cout<<"Database name changed... "<<file_name_<<std::endl;
	}


	std::string DataBase::getDataBaseName () {
		return file_name;
	}

	DataBase::~DataBase(){
		std::cout<<"Detroying database.."<<std::endl;
		file.close();
	}





	Classifier::Classifier () {

	}

	void Classifier::setImage(cv::Mat image){
		color_image=image;
		extracted_image=image;
	}

	void Classifier::imageExtractROI (cv::Mat im) {

		color_image=im;

		int half_side_length = 45;// Half side lenght= 50-(5)


		cv::Mat I (color_image, cv::Rect(cv::Point(color_image.cols/2-half_side_length, color_image.rows/2-half_side_length),
		cv::Point(color_image.cols/2+half_side_length, color_image.rows/2+half_side_length)) ); // using a rectangle

		extracted_image=I;

	}


	void Classifier::classify(){

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



		//selecting parameters for the HOG feature extractor. There are two feature extractor, one for each cell size (8x8, 16x16)
		hog.blockSize= cv::Size(16,16);
		hog.winSize= cv::Size(RES_X,RES_Y);
		hog.blockStride=cv::Size(8,8);
		hog.cellSize= cv::Size(8,8);

		hog2.blockSize= cv::Size(16,16);
		hog2.winSize= cv::Size(RES_X,RES_Y);
		hog2.blockStride=cv::Size(8,8);
		hog2.cellSize= cv::Size(16,16);

		//loading support vector machine
		SVM_scissors.load("svm_scissors");
		SVM_pen.load("svm_pen");
		SVM_pen_scissors.load("pen_scissors_svm");

		image= extracted_image;

		cv::resize(image, im, cv::Size(RES_X,RES_Y));
		im.copyTo(image, circle);

		//Computing the histogram of oriented gradients features

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

		//Evaluating the features in the binary classifiers
		classification_result1=  SVM_scissors.predict(features);
		classification_result2= SVM_pen.predict(features);
		classification_result3= SVM_pen_scissors.predict(features2);

		std::cout<<"Output of scissors classifier:"<<classification_result1<<std::endl;
		std::cout<<"Output of pen classifier:"<<classification_result2<<std::endl;
		std::cout<<"Output of scissors/pen classifier:"<<classification_result3<<std::endl;

		//Determining the class accordint to the result of the binary classifiers
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


	int Classifier::getClass() {
		return c;
	}

	Classifier::~Classifier(){

	}





	TrainingContainer::TrainingContainer () {
		training_data=cv::Mat::zeros(cv::Size(0,0), CV_32F);
		test_data=cv::Mat::zeros(cv::Size(0,0), CV_32F);
		training_labels=cv::Mat::zeros(cv::Size(0,0), CV_32F);
		test_labels=cv::Mat::zeros(cv::Size(0,0), CV_32F);
	}
	TrainingContainer::TrainingContainer (cv::Mat td, cv::Mat testd, cv::Mat tl, cv::Mat testl):
		training_data(td),
		test_data(testd),
		training_labels(tl),
		test_labels(testl){

	}


	void TrainingContainer::setData(cv::Mat td, cv::Mat testd, cv::Mat tl, cv::Mat testl){
		training_data= td;
		test_data= testd;
		training_labels= tl;
		test_labels= testl;

	}

	//Set of features adquired by HOG
	void TrainingContainer::setFeatures(cv::Mat f){
		features=f;
	}

	//Name of the file which contains the information of the learner
	void TrainingContainer::nameSVM(std::string n) {

		name_SVM=n;
	}

	cv::Mat TrainingContainer::getTrainingData(){
		return training_data;
	}

	cv::Mat TrainingContainer::getTestData(){
		return test_data;
	}

	cv::Mat TrainingContainer::getTrainingLabels(){
		return training_labels;
	}

	cv::Mat TrainingContainer::getTestLabels(){
		return test_labels;
	}

	//"Random vector" a function used to sort randomly a set of integer number. The integer number to be sorted are from 0 to m. It returns the vector with this numbers randomly sorted.
	std::vector <int> TrainingContainer::random_vector (int m) {

		std::vector <int> v_sorted, v;
		int x;

		for(int i=0; i<m; i++)
			v.push_back(i);

		for (int i=0; i< m; i++){
			srand(time(NULL));
			x= rand()%(int)(m-i);
			v_sorted.push_back(v.at(x));
			v.erase(v.begin()+x);
		}

		return v_sorted;

	}


	//Splits the data set in false positives and negatives
	void TrainingContainer::splitting(){

		std::vector<int> v, v_sorted;
		cv::Mat pos, neg;
		cv::Mat pos_training, neg_training;
		cv::Mat pos_test, neg_test;
		cv::Mat pos_training_labels, neg_training_labels;
		cv::Mat pos_test_labels, neg_test_labels;
		cv::Mat training, test;
		cv::Mat training_l, test_l;
		cv::Mat labels_training, labels_test;
		cv::FileStorage f;
		TrainingContainer training_info;
		int   x;


		std::cout<<"Features size:"<< features.size()<<std::endl;
		std::cout<<"Splitting samples in positives and negatives..."<<std::endl;

		//Splitting data in positives and negatives

		for(int i=0; i<features.rows ; i++){

			if(features.at<float>(i,features.cols-1)==1){//it is positive with labels
				pos.push_back(features.row(i).colRange(0,features.cols));
			}
			else{//it is negative
				neg.push_back(features.row(i).colRange(0,features.cols));
			}

		}

		std::cout<<"Spitting positives labels..."<<std::endl;


		v_sorted= random_vector(pos.rows);

		//Splitting positives labels in training and test
		for(int i=0; i<pos.rows; i++){

			x=v_sorted.at(i);

			if(i<TRAIN_RATIO*pos.rows){
				pos_training.push_back(pos.row(x).colRange(0, pos.cols-1));
				pos_training_labels.push_back(pos.at<float>(x,pos.cols-1));
			}
			else {
				pos_test.push_back(pos.row(x).colRange(0, pos.cols-1));
				pos_test_labels.push_back(pos.at<float>(x, pos.cols-1));

			}

		}

		v_sorted.clear();
		v.clear();

		//Spitting negatives labels in training and test
		std::cout<<"Splitting negatives labels..."<<std::endl;

		v_sorted= random_vector(neg.rows);
		for(int i=0; i<neg.rows; i++){

			x= v_sorted.at(i);

			if(i<TRAIN_RATIO*neg.rows){
				neg_training.push_back(neg.row(x).colRange(0, neg.cols-1));
				neg_training_labels.push_back(neg.at<float>(x,neg.cols-1));
			}
			else {
				neg_test.push_back(neg.row(x).colRange(0, neg.cols-1));
				neg_test_labels.push_back(neg.at<float>(x, neg.cols-1));

			}
		}



		//Concatenating data in a single matrix

		cv::vconcat(pos_training, neg_training, training);
		cv::vconcat(pos_training_labels, neg_training_labels, training_l);
		cv::vconcat(pos_test, neg_test, test);
		cv::vconcat(pos_test_labels, neg_test_labels, test_l);

		std::cout<<"Joining dataset..."<<std::endl;

		v_sorted.clear();
		v.clear();

		std::cout<<"Sorting training dataset..."<<std::endl;

		v_sorted= random_vector(training.rows);

		for(int i=0; i<training.rows; i++){

			x= v_sorted.at(i);
			training_data.push_back(training.row(x));
			training_labels.push_back(training_l.at<float>(x,0));
		}

		v_sorted.clear();
		v.clear();

		v_sorted= random_vector(test.rows);

		std::cout<<"Sorting test dataset..."<<std::endl;
		for(int i=0; i<test.rows; i++){

			x= v_sorted.at(i);
			test_data.push_back(test.row(x));
			test_labels.push_back(test_l.at<float>(x,0));
		}



		std::cout<<"Number of features:"<<training_data.cols;

		std::cout<<"Saving data for training..."<<std::endl;

		//saving everything
	//	f.open("HOG_Features.xml", cv::FileStorage::WRITE);
	//	f<<"Training_data" << training_data;
	//	f<<"Test_data"<< test_data;
	//	f<<"Labels_training_data"<< training_labels;
	//	f<<"Labels_test_data"<< test_labels;
	//	f.release();
	}

	//Save the SVM informaiton
	void TrainingContainer::saveSVM(){
		SVM.save(name_SVM.c_str());
		std::cout<<"Saved: "<<name_SVM.c_str()<<std::endl;
	}


	//Evaluates the performance of the learner on the training set and test set
	void TrainingContainer::evaluation() {


		cv::Mat training_classification (1,1,CV_32FC1), test_classification(1,1,CV_32FC1);
		CvANN_MLP neural_network;
		std::string ans;
		float training_error=0, test_error=0;
		float n_training_samples,n_test_samples,A,F=9,P=0;
		float false_positives_training=0, false_negatives_training=0, true_positives_training=0, true_negatives_training=0;
		float false_positives_test=0, false_negatives_test=0, true_positives_test=0, true_negatives_test=0;
		float rt, rv, exp;
		int n_features;


		std::cout<<"Loading data for training error evaluation..."<<std::endl;
		n_features= training_data.cols;
		n_training_samples= training_data.rows;


		//Calculating training error

		std::cout<<"Calculating training error..."<<std::endl;
		for(int i=0; i<n_training_samples; i++){


			//Evaluates the training data on the support vector machine
			rt= SVM.predict(training_data.row(i));


			exp= training_labels.at<float>(i,0);


			//Collecting information
			if(rt!=exp){
				training_error++;
			}


			if(exp==1 && rt==1)
				true_positives_training++;

			else if(exp==1 && rt==-1)
				false_negatives_training++;

			else if(exp==-1 && rt==-1)
				true_negatives_training++;

			else if (exp==-1 && rt==1){
				false_positives_training++;

			}
		}



		std::cout<<"True positives: "<<true_positives_training<<std::endl;
		std::cout<<"False positives: "<<false_positives_training<<std::endl;
		std::cout<<"True negatives: "<<true_negatives_training<<std::endl;
		std::cout<<"False negatives: "<<false_negatives_training<<std::endl;

		std::cout<<"*****************"<<std::endl;
		std::cout<<"Training Error:"<< training_error/n_training_samples<<std::endl;
		std::cout<<"*****************"<<std::endl;


		//Calculating Test error

		std::cout<<"Loading data for test error evaluation..."<<std::endl;
		n_test_samples= test_labels.rows;

		for(int i=0; i<n_test_samples; i++){


			rv= SVM.predict(test_data.row(i));
			exp= test_labels.at<float>(i,0);

			if(exp!=rv){

				test_error++;
			}

			if(exp==1 && rv==1)
				true_positives_test++;

			else if(exp==1 && rv==-1)
				false_negatives_test++;

			else if(exp==-1 && rv==-1)
				true_negatives_test++;

			else if (exp==-1 && rv==1)
				false_positives_test++;


		}

		//	Displaying information
		std::cout<<"True positives: "<<true_positives_test<<std::endl;
		std::cout<<"False positives: "<<false_positives_test<<std::endl;
		std::cout<<"True negatives: "<<true_negatives_test<<std::endl;
		std::cout<<"False negatives: "<<false_negatives_test<<std::endl;
		std::cout<<"Number of test labels: "<< n_test_samples<<std::endl;
		std::cout<<"*****************"<<std::endl;
		std::cout<<"Test Error:"<< test_error/n_test_samples<<std::endl;
		std::cout<<"*****************"<<std::endl;
		std::cout<<std::endl;

		std::cout<<"Size of training features vector:"<< training_data.size()<<std::endl;
		std::cout<<"Size of the training features vector before: "<<training_data.size()<<std::endl;

		std::cout<<"***********************************************"<<std::endl;
		std::cout<<" General information of classifier:          "<<std::endl;
		std::cout<<" Training                                    "<<std::endl;
		std::cout<<" PPV: "<<true_positives_training/(false_positives_training+true_positives_training)<<std::endl;
		std::cout<<" NPV: "<<true_negatives_training/(false_negatives_training+true_negatives_training)<<std::endl;
		std::cout<<" Sensivity: "<<true_positives_training/(true_positives_training+false_negatives_training)<<std::endl;
		std::cout<<" Specifity: "<<true_negatives_training/(true_negatives_training+false_positives_training)<<std::endl;

		std::cout<<" Test                                          "<<std::endl;
		std::cout<<" PPV: "<<true_positives_test/(false_positives_test+true_positives_test)<<std::endl;
		std::cout<<" NPV: "<<true_negatives_test/(false_negatives_test+true_negatives_test)<<std::endl;
		std::cout<<" Sensivity: "<<true_positives_test/(true_positives_test+false_negatives_test)<<std::endl;
		std::cout<<" Specifity: "<<true_negatives_test/(true_negatives_test+false_positives_test)<<std::endl;
		std::cout<<"***********************************************"<<std::endl;


	}


	//Train the SVM according to the features extracted and the labels.
	void TrainingContainer::training (){


		int n_features=0, n_training_samples=0, n_test_samples,a, n_neurons;
		std::string name_svm;

		std::cout<<"Loading data for training..."<<std::endl;
		n_training_samples= training_data.rows;
		n_features= training_data.cols;


		std::cout<<"Loading data for test error evaluation..."<<std::endl;
		n_test_samples= test_labels.rows;


		std::cout<<"Loaded samples... "<<std::endl;
		std::cout<<"Number of features: "<<n_features<<std::endl;
		std::cout<<"Number of training samples: "<<n_training_samples<<std::endl;


		//Set parameters of the SVM
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);
		params.degree =5;
		params.p=0.5;

		//Sets the C parameter determined by the user
		std::cout<<"Which value of C do you want to use?"<<std::endl;
		std::cin>>params.C;
		std::cout<<"Size of features:"<<training_data.size()<<std::endl;
		std::cout<<"Training the SVM..."<<std::endl;

		//Training
		SVM.train(training_data, training_labels, cv::Mat(), cv::Mat(), params);

		std::cout<<"Number of support vectors:"<<SVM.get_var_count()<<std::endl;
		std::cout<<"Support vector machine trained..."<<std::endl;
	}




	Preprocessing::Preprocessing () {
		std::cout<<"Preprocessor created..."<<std::endl;

	}

	void Preprocessing::readImage (std::string image_name_) {

		color_image= cv::imread(image_name_);
		if(color_image.data ==NULL) {
			std::cout<<"Error trying to load the image "<<image_name_<<std::endl;
			exit(0);
		}

		image_name= image_name_;
		std::cout<<"Image loaded..."<<std::endl;
		cv::namedWindow("Showing",1);
		cv::imshow("Showing", color_image);
	}


	cv::Mat Preprocessing::imageExtractROI () {


		if(color_image.data== NULL){
			std::cout<<"No image loaded. A image must be read first to use this method."<<std::endl;
			exit(0);
		}


		int half_side_length = 45;// Half side lenght= 50-(5)
		cv::Mat I (color_image, cv::Rect(cv::Point(color_image.cols/2-half_side_length, color_image.rows/2-half_side_length),
		cv::Point(color_image.cols/2+half_side_length, color_image.rows/2+half_side_length)) ); // using a rectangle
		extracted_image=I;

		std::cout<< "ROI Extracted..."<<std::endl;
		return I;
	}


	cv::Mat Preprocessing::getImage(){
		return color_image;
	}


