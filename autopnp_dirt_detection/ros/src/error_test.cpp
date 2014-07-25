#include <autopnp_dirt_detection/training.h>


int main () {


	DataBase db("Database.txt", FALSE);
	std::string name;
	int cl;
	float error=0;

	for(int i=0; i<db.getSize(); i++){
		Classifier c;

		name = db.getElement(i);
		std::cout<<"Analizying image "<<name<<" ..."<<std::endl;
		cv::Mat image= cv::imread(name+".jpg");


		std::cout<<"Image read..."<<std::endl;

		if(name[0]=='O' || name[0]=='P'){

			std::cout <<"ROI Extraction..."<<std::endl;
			std::cout<<"Image size: "<<image.size()<<std::endl;
			c.imageExtractROI(image);

		}
		else
			c.setImage(image);

		std::cout<<"Image set..."<<std::endl;
		c.classify();
		cl = c.getClass();

		if(name[0]=='P' && cl!=PEN) error++;
		else if(name[0]=='S' && cl!=SCISSORS) error++;
		else if (name[0]=='O' && cl!=NO_IDENTIFIED) error++;


	}

	std::cout<<"Error= "<<error/db.getSize()<<std::endl;




	return 0;
}
