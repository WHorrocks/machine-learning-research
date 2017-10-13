#include "TF1.h"
#include "TRandom3.h"
#include "TRandom1.h"
#include "TMath.h"
#include "TGraph.h"
#include <fstream>
#include <iostream>
#include <sstream>


void circleDataGenerator (int circle_Points, double circle_Sigma) {
	

	//Setting up the random number generators to generate the points.
	TRandom3 *gaussian_gen = new TRandom3();
	TRandom3 *theta_gen=new TRandom3(); 	
	
	//Creating the arrays to hold the points for the torus.	
	double Circle_x_array [circle_Points];
        double Circle_y_array[circle_Points];
        double Circle_radii_array[circle_Points];
        double Circle_theta_array[circle_Points];

	
	//Pathname of the directory to write the data to.
        string pathToDataDirectory="/Users/Will/Research/tensorflow/NewGraphs_8_27_17/";


	
	//Generating the points for the torus in polar coordinates. The radii are gaussian distributed centered around zero (because it is a circle). The angles are generated unifromly in the range [0,2PI].
	for(int i=0; i<circle_Points; i++) {
                Circle_radii_array[i] = gaussian_gen->Gaus(0,circle_Sigma);
                Circle_theta_array [i] = theta_gen->Uniform(0,TMath::TwoPi());
        }

	//Converting the previously generated polar coordinates (for the Torus) into cartesian points.
        for(int i =0; i<circle_Points; i++) {
                Circle_x_array [i] = (Circle_radii_array[i]*(TMath::Cos(Circle_theta_array[i])));
                Circle_y_array[i] = (Circle_radii_array[i] * (TMath::Sin(Circle_theta_array[i])));
        }



	//Writing the data in the format: xCoord,yCoord,1. It is written to a file that indicates the sigma of the circle. The 1 indicates that this point is a circle (used later for known results in ANN).
	ofstream outFile;
	std::ostringstream strs;
	strs << circle_Points;
        std::string FileName = strs.str();

	std::ostringstream strs2;
	strs2 << circle_Sigma;
	std::string SigmaString = strs2.str();
	FileName=FileName+"_"+SigmaString;
        FileName=pathToDataDirectory+FileName+".txt";
	outFile.open (FileName);

	for(int i=0; i<circle_Points; i=i+1) {
                outFile << Circle_x_array[i] << "," << Circle_y_array[i] << "," << 1 <<endl;
        }

	outFile.close();

	
}
