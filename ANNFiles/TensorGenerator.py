import tensorflow as tf
import csv
import numpy as np
import math


#Now the labels are a 1 by 2 matrix that with the 0th index indicating if the point is a circle and the 1st index indicating if the point is a torus (a 1 means the point is in a given group, a zero means it is not)
class TensorGenerator():

#The plan is I am going to store all of the data points into a big array and then shuffle that whole array so that way the right coords are with the right labels and then I am going to return that.

	def OutputData(self,CirclePath,TorusPath):

		CircleDataFileName=CirclePath
		TorusDataFileName=TorusPath

		TotalNumberOfPoints=0
		
		#Getting total number of Data Points
		with open(CircleDataFileName) as CircleData:
			for row in CircleData:
				TotalNumberOfPoints=TotalNumberOfPoints+1
		with open(TorusDataFileName) as TorusData:
			for row in TorusData:
				TotalNumberOfPoints=TotalNumberOfPoints+1

		
		RowNumber=0

		#Setting up the np array which I will cut up later to get the specific data points I need.
		
		#Adding the circle data points to the TotalData array
		TotalData=np.zeros([TotalNumberOfPoints,3])
		with open(CircleDataFileName) as CircleData:
			for row in CircleData:
				firstCommaIndex=row.index(',')
				secondString=row[firstCommaIndex+1:]
				secondCommaIndex=firstCommaIndex+secondString.index(',')



				#Reading in the x Coordinate (start of data to first comma)
				xCoord=row[0:firstCommaIndex]


				#Reading in the y Coordinate (after the first comma to right before the second comma)
				yCoord=row[firstCommaIndex+1:secondCommaIndex+1]


				#Reading in the last value (the label)
				label=row[secondCommaIndex+2:]


				TotalData[RowNumber,0]=xCoord
				TotalData[RowNumber,1]=yCoord
				TotalData[RowNumber,2]=label
				RowNumber=RowNumber+1		


		#Adding the Torus data points to the TotalData array
		with open(TorusDataFileName) as TorusData:
                        readTorus=csv.reader(TorusDataFileName, delimiter=",")
                        for row in TorusData:
                                firstCommaIndex=row.index(',')
                                secondString=row[firstCommaIndex+1:]
                                secondCommaIndex=firstCommaIndex+secondString.index(',')



                                #Reading in the x Coordinate (start of data to first comma)
                                xCoord=row[0:firstCommaIndex]


                                #Reading in the y Coordinate (after the first comma to right before the second comma)
                                yCoord=row[firstCommaIndex+1:secondCommaIndex+1]


                                #Reading in the last value (the label)
                                label=row[secondCommaIndex+2:]


                                TotalData[RowNumber,0]=xCoord
                                TotalData[RowNumber,1]=yCoord
                                TotalData[RowNumber,2]=label
                                RowNumber=RowNumber+1

		np.random.shuffle(TotalData)
		#Now TotalData is in a random order. That way when I iterate through it the data the ANN gets will be in a random order instead of going through all of the circles and then all of the Torus points.

		#Now cutting the array up into Coordinate and Label arrays to return
		
		#Coordinate arrays
		Coordinates=np.zeros([TotalNumberOfPoints,2])
		Num_Row=0

                for row in range(0,TotalNumberOfPoints):
			Coordinates[Num_Row,0]=TotalData[row,0]
			Coordinates[Num_Row,1]=TotalData[row,1]
			Num_Row=Num_Row+1


		
		#Labels array
		Labels=np.zeros([TotalNumberOfPoints,2])
		Num_Row=0
	
		for row in range(0,TotalNumberOfPoints):
			Labels[Num_Row,0]=TotalData[row,2]
			if TotalData[row,2]==1:
				Labels[Num_Row,1]=0
			else:
				Labels[Num_Row,1]=1
			Num_Row=Num_Row+1



		#Now setting up training and validation set.
                LineToEndTestSet=int(math.floor(0.8*TotalNumberOfPoints))
		Train_Coords=Coordinates[:LineToEndTestSet]
		Train_Labels=Labels[:LineToEndTestSet]

		Test_Coords=Coordinates[LineToEndTestSet:]
		Test_Labels=Labels[LineToEndTestSet:]


		return Train_Coords,Train_Labels,Test_Coords,Test_Labels,TotalNumberOfPoints
