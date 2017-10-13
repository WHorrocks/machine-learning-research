#include "TF1.h"
#include "TRandom3.h"
#include "TRandom1.h"
#include "TMath.h"
#include <fstream>
#include <iostream>
#include <sstream>

void torusDataGenV2 (int Torus_Points, double Torus_Radius, double Torus_Sigma) {
	

	//Setting up the random number generators to generate the points.
	TRandom3 *gaussian_gen = new TRandom3();
	TRandom3 *theta_gen=new TRandom3(); 	

	
	//Creating the arrays to hold the points for the torus.	
	double Torus_x_array [Torus_Points];
        double Torus_y_array[Torus_Points];
        double Torus_radii_array[Torus_Points];
        double Torus_theta_array[Torus_Points];

	
	//Pathname of the directory to write the data to.
        string pathToDataDirectory="/Users/Will/Research/tensorflow/NewGraphs_8_27_17/Data/ROOTGeneratedData2/TorusData/NoOverlap/";


	
	//Generating the points for the torus in polar coordinates. The radii are gaussian distributed centered around the user entered radius. The angles are generated unifromly in the range [0,2PI].
	for(int i=0; i<Torus_Points; i++) {
                Torus_radii_array[i] = gaussian_gen->Gaus(Torus_Radius,Torus_Sigma);
                Torus_theta_array [i] = theta_gen->Uniform(0,TMath::TwoPi());
        }

	//Converting the previously generated polar coordinates (for the Torus) into cartesian points.
        for(int i =0; i<Torus_Points; i++) {
                Torus_x_array [i] = (Torus_radii_array[i]*(TMath::Cos(Torus_theta_array[i])));
                Torus_y_array[i] = (Torus_radii_array[i] * (TMath::Sin(Torus_theta_array[i])));
        }



	//Writing the data in the format: xCoord,yCoord,0. It is written to a file that indicates the sigma of the sphere. The 0 indicates that this point is a torus (used later for known results in ANN).
	ofstream outFile;
	std::ostringstream strs;
	strs << Torus_Points;
	strs << "_";
        strs << Torus_Radius;
        std::string FileName = strs.str();

	std::ostringstream strs2;
	strs2 << Torus_Sigma;
	std::string SigmaString = strs2.str();
	FileName=FileName+"_"+SigmaString;
        FileName=pathToDataDirectory+FileName+".txt";
	outFile.open (FileName);

	for(int i=0; i<Torus_Points; i=i+1) {
                outFile << Torus_x_array[i] << "," << Torus_y_array[i] << "," << 0 <<endl;
        }

	outFile.close();

	
}
