
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <fstream>
#include <iostream>
#include<vector>


#define ATD at<double>
#define elif else if

using namespace cv;
using namespace std;

enum haar_type {
	horizontalFeature,
	verticalFeature,
	tripleFeature,
};
struct images {

	Mat image;
	bool face;
	double weight;// i will chose a threshold to optimize this value
};
struct myPair {

	double score;
	bool face;
	bool operator < (const myPair & str) const
	{
		return (score < str.score);
	}
};
struct feature {

	int function, polarity;
	pair<int, int> position;
	int width, height;
	double weight,threshold;
	vector<myPair>scores;
	haar_type type;
};
struct strongClassifier {

	vector<feature> features;
	double threshold;
	double FPR = 0;
	double DR = 0;
};
struct cascadeClassifier
{
	int stagesNumber = 0;
	double achivedFPR = 1.0;
	double achivedDR = 1.0;
	vector<strongClassifier> myClassifier;
};
double numbersOfPositive; //T+;
double numberOfNegative; //T-;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void read_LFW(vector<images> & imagesArray) {

	ifstream inFile;
	string location;
	inFile.open("C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessing Project Final/ImageProcessingTest/faceTest/pathImages.txt");
	if (!inFile.is_open())
	{
		cout << "error";
		return;
	}
	int count = 0;
	while (getline(inFile, location) && count < 1000)
	{
		Mat src = imread(location, IMREAD_GRAYSCALE);
		Mat dst;
		resize(src, dst, Size(24, 24), 0, 0, INTER_AREA);
		images singleImage;
		singleImage.face = 1;
		singleImage.image = dst;
		imagesArray.push_back(singleImage);
		count++;
	}
	cout << "Faces read =" << count << endl;

}
void read_CIFAR(vector<images> & imagesArray) {

	ifstream inFile;
	string location;
	inFile.open("C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessing Project Final/ImageProcessingTest/trainTest/pathImages.txt");
	if (!inFile.is_open())
	{
		cout << "error";
		return;
	}
	int count = 0;
	while (getline(inFile, location) && count < 2000)
	{
		Mat src = imread(location, IMREAD_GRAYSCALE);
		Mat dst;
		resize(src, dst, Size(24, 24), 0, 0, INTER_AREA);
		images singleImage;
		singleImage.face = 0;
		singleImage.image = dst;
		imagesArray.push_back(singleImage);
		count++;
	}
	cout << "Non-Faces read =" << count << endl;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<images> integralImage(vector<images>imagesArray)
{
	vector<images> integralImages;
	for (int i = 0; i < imagesArray.size(); i++)
	{
		images integralImage;
		integralImage.face = imagesArray[i].face;
		integral(imagesArray[i].image, integralImage.image);
		integralImages.push_back(integralImage);
	}
	//cout << "integralImages done'" << endl;
	return integralImages;
}
void InitializeWeights(int numberOfPositives, int numberOfNegatives, vector<images> & imagesArray)
{
	for (int i = 0; i < imagesArray.size(); i++)
	{
		if (imagesArray[i].face)
			imagesArray[i].weight = 1.0 / (2 * numberOfNegatives);
		else
			imagesArray[i].weight = 1.0 / (2 * numberOfPositives);
	}
}
Mat Haargenerator(int w, int h, haar_type type, bool reverse)//width number of columns height number of rows
{
	Mat haar_feature;
	int matrixElement;

	if (reverse)
		matrixElement = -1;
	else
		matrixElement = 1;
	if (type == horizontalFeature)
	{
		Mat haar_instance_white(h, w, CV_8UC1, Scalar(matrixElement));
		Mat haar_instance_black(h, w, CV_8UC1, Scalar(-matrixElement));
		hconcat(haar_instance_white, haar_instance_black, haar_feature);
	}
	else if (type == verticalFeature)
	{
		Mat haar_instance_white(h, w, CV_8UC1, Scalar(matrixElement));
		Mat haar_instance_black(h, w, CV_8UC1, Scalar(-matrixElement));
		vconcat(haar_instance_white, haar_instance_black, haar_feature);
	}
	else if (type == tripleFeature)
	{
		Mat haar_instance_white1(h, w, CV_8UC1, Scalar(matrixElement));
		Mat haar_instance_white2(h, w, CV_8UC1, Scalar(matrixElement));
		Mat haar_instance_black(h, w, CV_8UC1, Scalar(-matrixElement));
		Mat firstConcat;
		hconcat(haar_instance_white1, haar_instance_black, firstConcat);
		hconcat(firstConcat, haar_instance_white2, haar_feature);
	}
	return haar_feature;

}
double HaarCalculator(Mat image, pair<int, int>origin, int width, int height, haar_type type, bool reversed) // you determine the width and height of the filter with the dimension of 1 box only
{
	double convValue;
	Mat integralImage = image;
	origin.first++;
	origin.second++;
	if (width > image.cols || height > image.rows || origin.first < 1 || origin.second < 1)
		return 0; // not sure if i should return 0 or should i use better value to indicate out of index or throw exception maybe ?? 
	if (type == horizontalFeature)
	{
		pair<int, int>uLeft1, uRight1, lLeft1, lRight1; //first part of the filter
		pair<int, int>uLeft2, uRight2, lLeft2, lRight2; // second part of the filter

		uLeft1.first = origin.first - 1;
		uLeft1.second = origin.second - 1;

		uRight1.first = origin.first - 1;
		uRight1.second = origin.second + width - 1;

		lLeft1.first = origin.first + height - 1;
		lLeft1.second = origin.second - 1;

		lRight1.first = origin.first + height - 1;
		lRight1.second = origin.second + width - 1;

		uLeft2.first = origin.first - 1;
		uLeft2.second = origin.second + width - 1;

		uRight2.first = origin.first - 1;
		uRight2.second = origin.second + 2 * width - 1;

		lLeft2.first = origin.first + height - 1;
		lLeft2.second = origin.second + width - 1;

		lRight2.first = origin.first + height - 1;
		lRight2.second = origin.second + 2 * width - 1;

		double firstFilter = integralImage.at<int>(lRight1.first, lRight1.second) + integralImage.at<int>(uLeft1.first, uLeft1.second) - (integralImage.at<int>(lLeft1.first, lLeft1.second) + integralImage.at<int>(uRight1.first, uRight1.second));
		double secondFilter = integralImage.at<int>(lRight2.first, lRight2.second) + integralImage.at<int>(uLeft2.first, uLeft2.second) - (integralImage.at<int>(lLeft2.first, lLeft2.second) + integralImage.at<int>(uRight2.first, uRight2.second));
		// could use pointers to access better than .at 
		if (!reversed)
			convValue = firstFilter - secondFilter;
		else
			convValue = secondFilter - firstFilter;
	}
	if (type == verticalFeature)
	{
		pair<int, int>uLeft1, uRight1, lLeft1, lRight1; //first part of the filter
		pair<int, int>uLeft2, uRight2, lLeft2, lRight2; // second part of the filter

		uLeft1.first = origin.first - 1;
		uLeft1.second = origin.second - 1;

		uRight1.first = origin.first - 1;
		uRight1.second = origin.second + width - 1;

		lLeft1.first = origin.first + height - 1;
		lLeft1.second = origin.second - 1;

		lRight1.first = origin.first + height - 1;
		lRight1.second = origin.second + width - 1;

		uLeft2.first = origin.first + height - 1;
		uLeft2.second = origin.second - 1;

		uRight2.first = origin.first + height - 1;
		uRight2.second = origin.second + width - 1;

		lLeft2.first = origin.first + 2 * height - 1;
		lLeft2.second = origin.second - 1;

		lRight2.first = origin.first + 2 * height - 1;
		lRight2.second = origin.second + width - 1;

		/*cout << integralImage.at<double>(lRight1.first, lRight1.second)<<endl;
		cout << integralImage.at<double>(uLeft1.first, uLeft1.second) << endl;
		cout << integralImage.at<double>(lLeft1.first, lLeft1.second) << endl;
		cout << integralImage.at<double>(uRight1.first, uRight1.second) << endl;
		cout << integralImage.at<double>(lRight2.first, lRight2.second) << endl;
		cout << integralImage.at<double>(uLeft2.first, uLeft2.second) << endl;
		cout << integralImage.at<double>(lLeft2.first, lLeft2.second) << endl;
		cout << integralImage.at<double>(uRight2.first, uRight2.second) << endl;*/
		double firstFilter = integralImage.at<int>(lRight1.first, lRight1.second) + integralImage.at<int>(uLeft1.first, uLeft1.second) - (integralImage.at<int>(lLeft1.first, lLeft1.second) + integralImage.at<int>(uRight1.first, uRight1.second));
		double secondFilter = integralImage.at<int>(lRight2.first, lRight2.second) + integralImage.at<int>(uLeft2.first, uLeft2.second) - (integralImage.at<int>(lLeft2.first, lLeft2.second) + integralImage.at<int>(uRight2.first, uRight2.second));
		// could use pointers to access better than .at 
		if (!reversed)
			convValue = firstFilter - secondFilter;
		else
			convValue = secondFilter - firstFilter;
	}
	if (type == tripleFeature)
	{
		pair<int, int>uLeft1, uRight1, lLeft1, lRight1; //first part of the filter
		pair<int, int>uLeft2, uRight2, lLeft2, lRight2; // second part of the filter
		pair<int, int>uLeft3, uRight3, lLeft3, lRight3; // third part of the filter

		uLeft1.first = origin.first - 1;
		uLeft1.second = origin.second - 1;

		uRight1.first = origin.first - 1;
		uRight1.second = origin.second + width - 1;

		lLeft1.first = origin.first + height - 1;
		lLeft1.second = origin.second - 1;

		lRight1.first = origin.first + height - 1;
		lRight1.second = origin.second + width - 1;

		uLeft2.first = origin.first - 1;
		uLeft2.second = origin.second + width - 1;

		uRight2.first = origin.first - 1;
		uRight2.second = origin.second + 2 * width - 1;

		lLeft2.first = origin.first + height - 1;
		lLeft2.second = origin.second + width - 1;

		lRight2.first = origin.first + height - 1;
		lRight2.second = origin.second + 2 * width - 1;


		uLeft3.first = origin.first - 1;
		uLeft3.second = origin.second + 2 * width - 1;

		uRight3.first = origin.first - 1;
		uRight3.second = origin.second + 3 * width - 1;

		lLeft3.first = origin.first + height - 1;
		lLeft3.second = origin.second + 2 * width - 1;

		lRight3.first = origin.first + height - 1;
		lRight3.second = origin.second + 3 * width - 1;

		double firstFilter = integralImage.at<int>(lRight1.first, lRight1.second) + integralImage.at<int>(uLeft1.first, uLeft1.second) - (integralImage.at<int>(lLeft1.first, lLeft1.second) + integralImage.at<int>(uRight1.first, uRight1.second));
		double secondFilter = integralImage.at<int>(lRight2.first, lRight2.second) + integralImage.at<int>(uLeft2.first, uLeft2.second) - (integralImage.at<int>(lLeft2.first, lLeft2.second) + integralImage.at<int>(uRight2.first, uRight2.second));
		double thirdFilter = integralImage.at<int>(lRight3.first, lRight3.second) + integralImage.at<int>(uLeft3.first, uLeft3.second) - (integralImage.at<int>(lLeft3.first, lLeft3.second) + integralImage.at<int>(uRight3.first, uRight3.second));

		if (!reversed)
			convValue = firstFilter - secondFilter + thirdFilter;
		else
			convValue = secondFilter - firstFilter + thirdFilter;

		return convValue;
	}

	return convValue;
}
strongClassifier TrainStrongClassifier(vector<feature> & features, vector<images>imagesArray, int trainingTimes)
{
	strongClassifier myStrongClassifier;
	myStrongClassifier.threshold = 0;
	for (int t = 0; t < trainingTimes; t++)
	{
		double minError = INT_MAX;
		int featureIndex;
		double NormalizedSum = 0;
		for (int j = 0; j < imagesArray.size(); j++)
		{
			NormalizedSum += imagesArray[j].weight;
		}
		for (int j = 0; j < imagesArray.size(); j++)
		{
			imagesArray[j].weight /= NormalizedSum;
		}
		for (int i = 0; i < features.size(); i++)
		{
			double weightError = 0;
			for (int j = 0; j < imagesArray.size(); j++)
			{
				int faceDetected;
				int face = (imagesArray[j].face == 1) ? 1 : 0;
				double calculatedScore = HaarCalculator(imagesArray[j].image, features[i].position, features[i].width, features[i].height, features[i].type, 0);
				if (calculatedScore * features[i].polarity > features[i].threshold * features[i].polarity)
					faceDetected = 1;
				else
					faceDetected = 0;
				weightError += imagesArray[j].weight * abs(faceDetected - face);

			}
			if (weightError < minError)
			{
				minError = weightError;
				featureIndex = i;
				cout << "i am here initalizing featureIndex";
			}
		}
		for (int j = 0; j < imagesArray.size(); j++)
		{
			bool faceDetected;
			int face = (imagesArray[j].face == 1) ? 1 : 0;
			double calculatedScore = HaarCalculator(imagesArray[j].image, features[featureIndex].position, features[featureIndex].width, features[featureIndex].height, features[featureIndex].type, 0);
			if (calculatedScore * features[featureIndex].polarity > features[featureIndex].threshold * features[featureIndex].polarity)
				faceDetected = 1;
			else
				faceDetected = 0;
			int ei = (faceDetected == imagesArray[j].face) ? 0 : 1;
			double newWeight = imagesArray[j].weight* pow((minError / (1 - minError)), 1 - ei);
			imagesArray[j].weight = newWeight;
		}

		myStrongClassifier.threshold += log((1 - minError) / minError)/2;
		features[featureIndex].weight = log((1 - minError) / minError);
		myStrongClassifier.features.push_back(features[featureIndex]);
		features.erase(features.begin() + featureIndex);
		cout << "iteration  " << t << endl;
	}
	cout << "strong classifier trained";
	return myStrongClassifier;
}
void CreateFeatures(vector<feature> & featuresArray)
{
	const int frameSize = 24;
	const int featuresNumbers = 3;
	int count = 0;
	const int features[featuresNumbers][2] = { { 1,2 } ,{ 2,1 },{ 1,3 } }; // could add this to the range of filters{ 3, 1 }
	for (int i = 0; i < featuresNumbers; i++) {
		int sizeX = features[i][0];
		int sizeY = features[i][1];
		// Each position:
		for (int x = 0; x <= frameSize - sizeX; x++) {
			for (int y = 0; y <= frameSize - sizeY; y++) {
				// Each size fitting within the frameSize:
				for (int width = sizeY; width <= frameSize - y; width += sizeY) {
					for (int height = sizeX; height <= frameSize - x; height += sizeX) {
						count++;
						feature newfeature;
						newfeature.position.first = x;
						newfeature.position.second = y;
						if (i == 0)
						{
							newfeature.width = width / 2;
							newfeature.height = height;
							newfeature.type = horizontalFeature;
						}
						else if (i == 1)
						{
							newfeature.type = verticalFeature;
							newfeature.width = width;
							newfeature.height = height / 2;
						}
						else
						{
							newfeature.width = width / 3;
							newfeature.height = height;
							newfeature.type = tripleFeature;
						}

						featuresArray.push_back(newfeature);
					}
				}
			}
		}
	}

	cout << "features Created = " << count << endl;

}
void TrainWeakClassifierTest(vector<feature> & features, vector<images> imagesArray)
{
	myPair tempPair;
	for (int i = 0; i < features.size(); i++)
	{
		double positiveNumbers = 0; //	T+
		double negativeNumbers = 0;//	T-
		double previousPositiveNumbers = 0;//	S+
		double previousNegativeNumbers = 0;	//	S-	
		for (int j = 0; j < imagesArray.size(); j++)
		{
			double calculatedScore = HaarCalculator(imagesArray[j].image, features[i].position, features[i].width, features[i].height, features[i].type, 0);
			tempPair.score = calculatedScore;
			tempPair.face = imagesArray[j].face;
			features[i].scores.push_back(tempPair);
			if (imagesArray[j].face)
				positiveNumbers++;
			else
				negativeNumbers++;

		}
		sort(features[i].scores.begin(), features[i].scores.end());
		double minError = INT_MAX;
		double currentError;
		for (int j = 0; j < imagesArray.size(); j++)
		{
			currentError = min(previousPositiveNumbers + (negativeNumbers - previousNegativeNumbers), previousNegativeNumbers + (positiveNumbers - previousPositiveNumbers));
			if (currentError < minError)
			{
				minError = currentError;
				features[i].threshold = features[i].scores[j].score;

				if (previousPositiveNumbers < positiveNumbers - previousPositiveNumbers)
					features[i].polarity = 1;
				else
					features[i].polarity = -1;

			}

			if (features[i].scores[j].face == 0)
				previousNegativeNumbers++;
			else
				previousPositiveNumbers++;
		}
		cout << " features i =  " << i << endl;
	}
}
void SaveFeatures(vector<feature> features, string location)
{
	ofstream outFile;
	outFile.open(location);
	for (int i = 0; i < features.size(); i++)
	{
		outFile << features[i].position.first << " " << features[i].position.second << " " << features[i].width << " " << features[i].height << " " << features[i].type << " "
			<< features[i].threshold << " " << features[i].polarity << endl;
	}
	outFile.close();
}
void SaveStrongClassifier(strongClassifier myStrongClassifier, string location)
{
	ofstream outFile;
	outFile.open(location);
	outFile << myStrongClassifier.features.size() << endl;
	for (int i = 0; i < myStrongClassifier.features.size(); i++)
	{
		outFile << myStrongClassifier.features[i].position.first << " " << myStrongClassifier.features[i].position.second << " " << myStrongClassifier.features[i].width << " " << myStrongClassifier.features[i].height << " " << myStrongClassifier.features[i].type << " "
			<< myStrongClassifier.features[i].threshold << " " << myStrongClassifier.features[i].polarity << " "<< myStrongClassifier.features[i].weight << endl;
	}
	outFile << myStrongClassifier.threshold;
	outFile.close();
	cout << "Strong Classifier Trained";
}
vector<feature>LoadFeatures(string location)
{
	vector<feature> features;
	ifstream inFile;
	string line;
	inFile.open(location);
	while (getline(inFile, line))
	{
		feature singleFeature;
		stringstream lineStream(line);
		int value;
		int type;
		lineStream >> singleFeature.position.first >> singleFeature.position.second >> singleFeature.width >> singleFeature.height >> type
			>> singleFeature.threshold >> singleFeature.polarity;
		if (type == 0)
			singleFeature.type = horizontalFeature;
		else if (type == 1)
			singleFeature.type = verticalFeature;
		else
			singleFeature.type = tripleFeature;
		features.push_back(singleFeature);
	}
	cout << "weak features loaded" << endl;
	return features;
}
vector<int> Predict(cascadeClassifier Classifier, vector<images> testSet)
{
	vector<int> indecies;
	for (int q = 0; q < Classifier.stagesNumber; q++)
	{
		vector<images>newTestSet;
		vector<int> newIndecies;
		for (int i = 0; i < testSet.size(); i++)
		{
			double predict = 0;
			for (int j = 0; j < Classifier.myClassifier[q].features.size(); j++)
			{
				double calculatedValue = HaarCalculator(testSet[i].image, Classifier.myClassifier[q].features[j].position, Classifier.myClassifier[q].features[j].width, Classifier.myClassifier[q].features[j].height, Classifier.myClassifier[q].features[j].type, 0);
				predict += Classifier.myClassifier[q].features[j].weight* (calculatedValue* Classifier.myClassifier[q].features[j].polarity >  Classifier.myClassifier[q].features[j].polarity *  Classifier.myClassifier[q].features[j].threshold);
			}
			if (predict > Classifier.myClassifier[q].threshold && q == 0)
			{
				newIndecies.push_back(i);
				newTestSet.push_back(testSet[i]);
			}
			else if ((predict > Classifier.myClassifier[q].threshold) && q != 0)
			{
				newIndecies.push_back(indecies[i]);
				newTestSet.push_back(testSet[i]);
			}
		}
		testSet = newTestSet;
		indecies = newIndecies;
	}

	return indecies;
}
pair <double, double> Evaluate(int numPositive, int imageLength, vector<int> indices, double & cascadeFPR, double & cascadeDR)
{
	double fpr = 0;
	double dr = 0;
	for (int i = 0; i < indices.size(); i++)
	{
		if (indices[i] >= numPositive)
			fpr += indices[i];
		else
			dr += indices[i];
	}
	fpr = fpr / (imageLength - numPositive);
	dr = dr / numPositive;
	pair <double, double> one;
	one = make_pair(fpr, dr);
	return one;
}
void CascadeClassifier(vector<feature>features, double fprFinal, double drFinal, vector<images>testSet, vector<images>imagesArray)
{
	cascadeClassifier myClassifier;
	double fprRatio = 0.5;
	int numPositive = 0;
	double drRatio = 0.1;
	vector<images>integralImagesArray = integralImage(imagesArray);
	vector<images>integraltestSet = integralImage(testSet);
	for (int i = 0; i < testSet.size(); i++)
	{
		if (testSet[i].face)
			numPositive++;
	}
	strongClassifier newStage;
	while (myClassifier.achivedFPR > fprFinal)
	{
		vector<feature>cpyFeatures = features;
		myClassifier.stagesNumber++;
		double cascadeFPR = myClassifier.achivedFPR;
		double cascadeDR = myClassifier.achivedDR;
		double newStageGoalFPR = fprRatio* myClassifier.achivedFPR;
		double newStageGoalDR = drFinal;
		while (cascadeFPR > newStageGoalFPR)
		{
			int numOfWeakClassifiers = int(3 + 5 * log10(myClassifier.stagesNumber));
			cout << "Training String Classifier with this many weak classifiers " << numOfWeakClassifiers << endl;
			newStage = TrainStrongClassifier(cpyFeatures, integralImagesArray, numOfWeakClassifiers);
			double thr = abs(newStage.threshold);

			while (true)
			{
				vector<int> Int = Predict(myClassifier, testSet);
				pair<double, double> One = Evaluate(numPositive, testSet.size(), Int, cascadeFPR, cascadeDR);
				cascadeDR = One.second;
				cascadeFPR = One.first;
				if (cascadeDR > drFinal)
					break;
				newStage.threshold -= 0.1*thr;
			}
		}
		newStage.FPR = cascadeFPR;
		newStage.DR = cascadeDR;
		myClassifier.achivedDR = cascadeDR;
		myClassifier.achivedFPR = cascadeFPR;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
strongClassifier loadStrongClassifier(string location)
{
	strongClassifier myStrongClassifier;
	ifstream inFile;
	string line;
	inFile.open(location);
	if (!inFile.is_open())
	{
		cout << "error";
	}
	getline(inFile, line);
	stringstream lineStream(line);
	int x;
	lineStream >> x;
	while (x > 0)
	{
		getline(inFile, line);
		feature singleFeature;
		stringstream lineStream(line);
		int value;
		int type;
		lineStream >> singleFeature.position.first >> singleFeature.position.second >> singleFeature.width >> singleFeature.height >> type
			>> singleFeature.threshold >> singleFeature.polarity >> singleFeature.weight;
		if (type == 0)
			singleFeature.type = horizontalFeature;
		else if (type == 1)
			singleFeature.type = verticalFeature;
		else
			singleFeature.type = tripleFeature;
		myStrongClassifier.features.push_back(singleFeature);
		x--;
	}
	getline(inFile, line);
	stringstream stream(line);
	stream >> myStrongClassifier.threshold;
	cout << "strong Classifier Loaded" << endl;
	return myStrongClassifier;
}
void testCamera(cascadeClassifier myClassifier)
{
	VideoCapture cap;
	if (!cap.open(0))
		return;
	vector<images> imagesArray;
	images imageTemp;
	while (1)
	{
		Mat frame,crp1,crp2,crp3,crp4;
		cap >> frame;
		/*int x1, x2, x3, x4, y1, y2, y3, y4, width, height;
		x1 = y1 = 0;
		x2 = 320; y2 = 0;
		x3 = 0; y3 = 240;
		x4 = 320; y4 = 240;
		width = 320;
		height = 240;
		Mat ROI1(frame, Rect(x1, y1, width, height));
		ROI1.copyTo(crp1);
		imageTemp.image = crp1;
		imagesArray.push_back(imageTemp);
		Mat ROI2(frame, Rect(x2, y2, width, height));
		ROI2.copyTo(crp2);
		imageTemp.image = crp2;
		imagesArray.push_back(imageTemp);
		Mat ROI3(frame, Rect(x3, y3, width, height));
		ROI3.copyTo(crp3);
		imageTemp.image = crp3;
		imagesArray.push_back(imageTemp);
		Mat ROI4(frame, Rect(x4, y4, width, height));
		ROI4.copyTo(crp4);
		imageTemp.image = crp4;*/
		cv::Mat greyMat;
		cv::cvtColor(frame, greyMat, cv::COLOR_BGR2GRAY);
		Mat ROI(greyMat, Rect(80, 0, 480, 480));
		Mat croppedFrame;
		ROI.copyTo(croppedFrame);
		imageTemp.image = croppedFrame;
		imagesArray.push_back(imageTemp);
		vector<images> resizedImages;
		for (int i = 0; i < imagesArray.size(); i++)
		{	
			Mat dst;
			resize(imagesArray[i].image, dst, Size(24, 24), 0, 0, INTER_AREA);
			images resizedImage;
			resizedImage.image = dst;
			resizedImages.push_back(resizedImage);
		}
		imshow("camera", resizedImages[0].image);
		vector<images>integralImagesArray = integralImage(resizedImages);
		vector<int> indecies = Predict(myClassifier, integralImagesArray);
		cout << indecies.size();
		if (frame.empty())
			break;
		if (indecies.size() != 0)
			imshow("this is you, smile!:)", imagesArray[indecies[0]].image);
		else
			imshow("this is you, smile!:)",frame);

		if (waitKey(10) == 27)
			break;
		imagesArray.clear();
		indecies.clear();
	}
	return;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
	//vector<images>imagesArray;
	//vector<feature> features;
	//CreateFeatures(features);
	//read_CIFAR(imagesArray);
	//read_LFW(imagesArray);
	//vector<images>integralImagesArray = integralImage(imagesArray);
	//TrainWeakClassifierTest(features, integralImagesArray);
	//SaveFeatures(features, "C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessingTest/ImageProcessingTest/weakFeatures.txt");
	//InitializeWeights(50, 100, integralImagesArray);
	//features = LoadFeatures("C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessingTest/ImageProcessingTest/weakFeatures.txt");
	//strongClassifier myStrongClassifier = TrainStrongClassifier(features, integralImagesArray, 20);
	//SaveStrongClassifier(myStrongClassifier,"C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessingTest/ImageProcessingTest/strongClassifier.txt");
	//double fprFinal = 0.0009;
	//double drFinal = 0.95;
	//CascadeClassifier(features, fprFinal, drFinal, testSet, imagesArray);
	

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	vector<images> testSet;
	read_LFW(testSet);
	/*read_CIFAR(testSet);
	vector<images>integralTestSet = integralImage(testSet);
	strongClassifier myStrongClassifier = loadStrongClassifier("C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessing Project Final/ImageProcessingTest/strongClassifier.txt");
	cascadeClassifier myCascadeClassifier;
	myCascadeClassifier.myClassifier.push_back(myStrongClassifier);
	myCascadeClassifier.stagesNumber = 1;
	vector<int> indices = Predict(myCascadeClassifier, integralTestSet);
	int truePositive = 0, falsePositive = 0, falseNegative = 0, trueNegative = 0 ;
	for (int i = 0; i < indices.size(); i++)
	{
		if (indices[i] < 1000)
			truePositive++;
		else
			falsePositive++;
	}
	falseNegative = 1000 - truePositive;
	trueNegative = 2000 - falsePositive;
	cout << endl;
	cout << "TruePositive = " << truePositive << "		 " << "FalsePositive = " << falsePositive << endl;
	cout << "FalseNegative = " << falseNegative << "		" << "TrueNegative = " << trueNegative << endl;
	cout << "Recall = " << truePositive / (1.0 *(truePositive + falseNegative)) << "		" << " Percision = " << truePositive / (1.0 * (truePositive + falsePositive));*/

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	strongClassifier myStrongClassifier = loadStrongClassifier("C:/Users/Owner/Documents/Visual Studio 2015/Projects/ImageProcessing Project Final/ImageProcessingTest/strongClassifier.txt");
	cascadeClassifier myCascadeClassifier;
	myCascadeClassifier.myClassifier.push_back(myStrongClassifier);
	myCascadeClassifier.stagesNumber = 1;
	imshow("picture", testSet[0].image);
	testCamera(myCascadeClassifier);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int n;
	cin >> n;

}
