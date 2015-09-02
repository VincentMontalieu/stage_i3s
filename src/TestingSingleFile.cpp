#include "Soft.hpp"
#include <fstream>
#include <stdio.h>
#include <utility>
#include <math.h>
#include <chrono>
#include <map>
#include <algorithm>
#include <functional>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace cv::xfeatures2d;
using namespace cv::ml;

/***** Outils de détection et d'extraction de features *****/

Ptr<SiftFeatureDetector> featureDetector;
Ptr<SiftDescriptorExtractor> descExtractor;
Ptr<SoftBOWImgDescriptorExtractor> bowExtractor;
Ptr<DescriptorMatcher> descMatcher;

// Le chemin vers le dossier principal contenant les données, celui vers le fichier training.data ainsi que celui vers l'image client
string data_directory, training_data, uploaded_image;

// Le chemin vers le fichier de log
string log_file;

// Le nombre de clusters à utiliser
int nbr_cluster;

// Représente l'ensemble des classes (plantes) présentes dans la base d'apprentisage
vector<string> training_classes;

// La matrice qui contient le vocabulaire principal
Mat vocabulary;

// Le taux d'erreur toléré pour le SVM
double c;

// Le vecteur contenant tous les SVM chargés
vector<Ptr<SVM>> svms;

// Le vecteur contenant les paires Espèce / Score
vector<pair<string, float>> predictions;

/**** Méthodes ****/

void getClasses();
void loadVocabulary();
void setVocabulary();
Mat calcDescriptor();
void loadSVM();
void testSVM(Mat current_descriptor);
void computePredictResults();
void renderJSON();

// Permet d'effectuer le tri du vecteur Espèces / Scores selon le score
struct sort_pred {
	bool operator()(const pair<string, float> &left, const pair<string, float> &right) {
		return left.second > right.second;
	}
};

void getClasses()
{
	/**** TRAINING.DATA ****/

	FILE *in;

	if ((in = fopen(training_data.c_str(), "rt")) != NULL)
	{
		char buffer[100];

		while (read_line(in, buffer, sizeof buffer))
		{
			vector<string> line;
			line = parseLine(buffer);

			if (find(training_classes.begin(), training_classes.end(), line[1]) == training_classes.end())
			{
				training_classes.push_back(line[1]);
			}
		}
	}

	else
	{
		cout << "Probleme fichier : " << training_data << endl;
		exit(-1);
	}

	fclose(in);
}

void loadVocabulary()
{
	cout << "Using " << nbr_cluster << " clusters" << endl << endl;
	cout << "Loading Vocabulary ...." << endl;
	string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string((long long)nbr_cluster) + ".xml.gz";
	vocabulary = loadBOWDescriptor(vocabulary_file_path, "vocabulary");
	cout << "Vocabulary loaded" << endl << endl;
}

void setVocabulary()
{
	cout << "Setting Vocabulary ...." << endl;
	bowExtractor->setVocabulary(vocabulary);
	cout << "Vocabulary set" << endl << endl;
}

Mat calcDescriptor()
{
	vector<KeyPoint> imageKeypoints;
	Mat bowDescriptor;

	Mat colorImage = imread(uploaded_image);

	if (!colorImage.cols)
	{
		cout << "Error while opening file " << uploaded_image << endl;
	}

	else
	{
		cout << "Loading image: " << uploaded_image << endl;
	}

	featureDetector->detect(colorImage, imageKeypoints);
	bowExtractor->compute(colorImage, imageKeypoints, bowDescriptor, 5);

	if (!bowDescriptor.cols)
	{
		cout << "Error while computing BOW histogram for file " << uploaded_image << endl ;
	}

	else
	{
		cout << "Computing histogram" << endl ;
	}

	return bowDescriptor;
}

void loadSVM()
{
	for (size_t i = 0; i < training_classes.size(); i++)
	{
		string svm_to_load = data_directory + PLANTS_SVM_FOLDER + "svm:" + training_classes[i] + "." + to_string((long long)nbr_cluster) + "." + to_string((long long)c) + ".xml.gz";
		cout << "Loading SVM: " << svm_to_load << endl;
		Ptr<SVM> svm = Algorithm::load<SVM>(svm_to_load);
		svms.push_back(svm);
	}

	cout << endl;
}

void testSVM(Mat bowDescriptors)
{
	cout << "SVM prediction" << endl;

	string prediction;
	float score;

	for (size_t svm_index = 0; svm_index < svms.size(); svm_index++)
	{
		float signMul = 1.f;

		float classVal = svms[svm_index]->predict(bowDescriptors);
		float scoreVal = svms[svm_index]->predict(bowDescriptors, noArray(), StatModel::Flags::RAW_OUTPUT);

		signMul = (classVal < 0) == (scoreVal < 0) ? 1.f : -1.f;
		score = signMul * scoreVal;

		prediction = training_classes[svm_index];

		predictions.push_back(pair<string, float>(prediction, score));
	}

	sort(predictions.begin(), predictions.end(), sort_pred());

	cout << endl;
}

void computePredictResults()
{
	cout << endl << "********* TESTING IMAGES *********" << endl << endl;

	Mat current_descriptor;
	string image_to_open;

	loadVocabulary();
	setVocabulary();
	loadSVM();

	current_descriptor = calcDescriptor();
	testSVM(current_descriptor);

	renderJSON();
}

void renderJSON()
{
	ofstream out;
	out.open(log_file, ios::out | ios::app);

	out << "[ ";

	if (predictions.size() > 10)
	{
		for (size_t i = 0; i < 9; i++)
		{
			out << "{" << "\"class\": \"" << predictions[i].first << "\",\"score\": \"" << predictions[i].second << "\"},";
		}

		out << "{" << "\"class\": \"" << predictions[9].first << "\",\"score\": \"" << predictions[9].second << "\"}]";
	}

	else
	{
		for (size_t i = 0; i < predictions.size() - 1; i++)
		{
			out << "{" << "\"class\": \"" << predictions[i].first << "\",\"score\": \"" << predictions[i].second << "\"},";
		}

		out << "{" << "\"class\": \"" << predictions[predictions.size() - 1].first << "\",\"score\": \"" << predictions[predictions.size() - 1].second << "\"}]";
	}

	out.close();
}

void help(char* argv[])
{
	cout << "Usage: " << argv[0] << " Data_folder Nbr_cluster C Image_to_analyze Log_file" << endl;
}

int main(int argc, char* argv[])
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	if (argc != 6)
	{
		help(argv);
		exit(-1);
	}

	// detecteur de POI
	featureDetector = SIFT::create();

	// descripteur de POI
	descExtractor = SIFT::create();

	if (featureDetector.empty() || descExtractor.empty())
	{
		cout << "Error in detector and extractor";
		exit(-1);
	}

	descMatcher = DescriptorMatcher::create("BruteForce");

	if (descMatcher.empty())
	{
		cout << "probleme matcher";
	}

	/*** Generateur de BOW ***/
	bowExtractor = new SoftBOWImgDescriptorExtractor(descExtractor, descMatcher); // SOFT ASSIGNMENT

	data_directory = argv[1];
	nbr_cluster = atoi(argv[2]);
	c = atof(argv[3]);
	uploaded_image = argv[4];
	log_file = argv[5];

	remove(log_file.c_str());

	setDataDirectoryPath(data_directory);
	training_data = data_directory + TRAINING_DATA_FILE;

	getClasses();
	computePredictResults();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

	cout << "TESTING TIME: " << convertTime(duration) << endl;

	return 0;
}
