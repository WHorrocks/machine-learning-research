import tensorflow as tf
import csv
import numpy as np
import math


#This makes np arrays that will be used by the ANN. They are split into 8000 2000 groups (for Circle and Torus), and each have their own set of labels associated with them. 
#Now the labels are a 1 by 2 matrix that with the 0th index indicating if the point is a circle and the 1st index indicating if the point is a torus (a 1 means the point is in a given group, a zero means it is not)
class UnshuffledDataGenerator():


	def OutputData(self):

		CircleDataFileName_2000Points="/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ROOTGeneratedData/CircleData/Circle_2000_Points.txt"
		CircleDataFileName_8000Points="/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ROOTGeneratedData/CircleData/Circle_8000_Points.txt"


		TorusDataFileName_2000Points="/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ROOTGeneratedData/TorusData/SignificantOverlap/SignificantOverlap_2000.txt"
                TorusDataFileName_8000Points="/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ROOTGeneratedData/TorusData/SignificantOverlap/SignificantOverlap_8000.txt"


		
		RowNumber=0

		#Setting up the circle np arrays which will hold the Circle points.	
		Circle_Array_2000=np.zeros([2000,3])
		Circle_Array_8000=np.zeros([8000,3])

		#Setting up the Torus np arrays which will hold the Torus points
		Torus_Array_2000=np.zeros([2000,3])
		Torus_Array_8000=np.zeros([8000,3])

		#Writing the 2000 Circle points into an array
		with open(CircleDataFileName_2000Points) as CircleData_2000:
			for row in CircleData_2000:
				firstCommaIndex=row.index(',')
				secondString=row[firstCommaIndex+1:]
				secondCommaIndex=firstCommaIndex+secondString.index(',')



				#Reading in the x Coordinate (start of data to first comma)
				xCoord=row[0:firstCommaIndex]


				#Reading in the y Coordinate (after the first comma to right before the second comma)
				yCoord=row[firstCommaIndex+1:secondCommaIndex+1]


				#Reading in the last value (the label)
				label=row[secondCommaIndex+2:]


				Circle_Array_2000[RowNumber,0]=xCoord
				Circle_Array_2000[RowNumber,1]=yCoord
				Circle_Array_2000[RowNumber,2]=label
				RowNumber=RowNumber+1	
		RowNumber=0
		#Writing the 8000 Circle points into an array
		with open(CircleDataFileName_8000Points) as CircleData_8000:
                        for row in CircleData_8000:
                                firstCommaIndex=row.index(',')
                                secondString=row[firstCommaIndex+1:]
                                secondCommaIndex=firstCommaIndex+secondString.index(',')



                                #Reading in the x Coordinate (start of data to first comma)
                                xCoord=row[0:firstCommaIndex]


                                #Reading in the y Coordinate (after the first comma to right before the second comma)
                                yCoord=row[firstCommaIndex+1:secondCommaIndex+1]


                                #Reading in the last value (the label)
                                label=row[secondCommaIndex+2:]


                                Circle_Array_8000[RowNumber,0]=xCoord
                                Circle_Array_8000[RowNumber,1]=yCoord
                                Circle_Array_8000[RowNumber,2]=label
                                RowNumber=RowNumber+1
		
		RowNumber=0
		#Writing the 2000 Torus points into an array
                with open(TorusDataFileName_2000Points) as TorusData_2000:
                        for row in TorusData_2000:
                                firstCommaIndex=row.index(',')
                                secondString=row[firstCommaIndex+1:]
                                secondCommaIndex=firstCommaIndex+secondString.index(',')



                                #Reading in the x Coordinate (start of data to first comma)
                                xCoord=row[0:firstCommaIndex]


                                #Reading in the y Coordinate (after the first comma to right before the second comma)
                                yCoord=row[firstCommaIndex+1:secondCommaIndex+1]


                                #Reading in the last value (the label)
                                label=row[secondCommaIndex+2:]


                                Torus_Array_2000[RowNumber,0]=xCoord
                                Torus_Array_2000[RowNumber,1]=yCoord
                                Torus_Array_2000[RowNumber,2]=label
                                RowNumber=RowNumber+1		


		RowNumber=0
                #Writing the 8000 Torus points into an array
                with open(TorusDataFileName_8000Points) as TorusData_8000:
                        for row in TorusData_8000:
                                firstCommaIndex=row.index(',')
                                secondString=row[firstCommaIndex+1:]
                                secondCommaIndex=firstCommaIndex+secondString.index(',')



                                #Reading in the x Coordinate (start of data to first comma)
                                xCoord=row[0:firstCommaIndex]


                                #Reading in the y Coordinate (after the first comma to right before the second comma)
                                yCoord=row[firstCommaIndex+1:secondCommaIndex+1]


                                #Reading in the last value (the label)
                                label=row[secondCommaIndex+2:]


                                Torus_Array_8000[RowNumber,0]=xCoord
                                Torus_Array_8000[RowNumber,1]=yCoord
                                Torus_Array_8000[RowNumber,2]=label
                                RowNumber=RowNumber+1




		#Now cutting the arrays up into Coordinate and Label arrays to return
		
		#Coordinate arrays

		#Circle Coordinate Arrays
		Circle_Coordinates_2000=np.zeros([2000,2])
		Circle_Coordinates_8000=np.zeros([8000,2])


		#Torus Coordinate Arrays
                Torus_Coordinates_2000=np.zeros([2000,2])
                Torus_Coordinates_8000=np.zeros([8000,2])



		#Making the 2000 Circle Coordinate Array
		Num_Row=0

                for row in range(0,2000):
			Circle_Coordinates_2000[Num_Row,0]=Circle_Array_2000[row,0]
			Circle_Coordinates_2000[Num_Row,1]=Circle_Array_2000[row,1]
			Num_Row=Num_Row+1


		#Making the 8000 Circle Coordinate Array
                Num_Row=0

                for row in range(0,8000):
                        Circle_Coordinates_8000[Num_Row,0]=Circle_Array_8000[row,0]
                        Circle_Coordinates_8000[Num_Row,1]=Circle_Array_8000[row,1]
                        Num_Row=Num_Row+1



		#Making the 2000 Torus Coordinate Array
                Num_Row=0

                for row in range(0,2000):
                        Torus_Coordinates_2000[Num_Row,0]=Torus_Array_2000[row,0]
                        Torus_Coordinates_2000[Num_Row,1]=Torus_Array_2000[row,1]
                        Num_Row=Num_Row+1


		#Making the 8000 Torus Coordinate Array
                Num_Row=0

                for row in range(0,8000):
                        Torus_Coordinates_8000[Num_Row,0]=Torus_Array_8000[row,0]
                        Torus_Coordinates_8000[Num_Row,1]=Torus_Array_8000[row,1]
                        Num_Row=Num_Row+1


		
		#Labels array


		#Circle Arrays
		Circle_Labels_2000=np.zeros([2000,2])
		Circle_Labels_8000=np.zeros([8000,2])

		#Torus Arrays
		Torus_Labels_2000=np.zeros([2000,2])
                Torus_Labels_8000=np.zeros([8000,2])
		
	

		#2000 Circle Labels
		Num_Row=0
	
		for row in range(0,2000):
			Circle_Labels_2000[Num_Row,0]=Circle_Array_2000[row,2]
			if Circle_Array_2000[row,2]==1:
				Circle_Labels_2000[Num_Row,1]=0
			else:
				Circle_Labels_2000[Num_Row,1]=1
			Num_Row=Num_Row+1


		#8000 Circle Labels
                Num_Row=0

                for row in range(0,8000):
                        Circle_Labels_8000[Num_Row,0]=Circle_Array_8000[row,2]
                        if Circle_Array_8000[row,2]==1:
                                Circle_Labels_8000[Num_Row,1]=0
                        else:
                                Circle_Labels_8000[Num_Row,1]=1
                        Num_Row=Num_Row+1


		#2000 Torus Labels
                Num_Row=0

                for row in range(0,2000):
                        Torus_Labels_2000[Num_Row,0]=Torus_Array_2000[row,2]
                        if Torus_Array_2000[row,2]==1:
                                Torus_Labels_2000[Num_Row,1]=0
                        else:
                                Torus_Labels_2000[Num_Row,1]=1
                        Num_Row=Num_Row+1


		#8000 Torus Labels
                Num_Row=0

                for row in range(0,8000):
                        Torus_Labels_8000[Num_Row,0]=Torus_Array_8000[row,2]
                        if Torus_Array_8000[row,2]==1:
                                Torus_Labels_8000[Num_Row,1]=0
                        else:
                                Torus_Labels_8000[Num_Row,1]=1
                        Num_Row=Num_Row+1
		

		return Circle_Coordinates_2000,Circle_Labels_2000,Circle_Coordinates_8000,Circle_Labels_8000,Torus_Coordinates_2000,Torus_Labels_2000,Torus_Coordinates_8000,Torus_Labels_8000
