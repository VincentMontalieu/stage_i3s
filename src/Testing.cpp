#include "Soft.hpp"
#include <fstream>
#include <stdio.h>
#include <utility>
#include <math.h>
#include <chrono>

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

// Le chemin vers le dossier principal contenant les données ainsi que celui vers le fichier training.data
string data_directory, training_data, testing_data;

// Le nombre de clusters à utiliser
int nbr_cluster;

// Représente l'ensemble des classes (plantes) présentes dans la base d'apprentisage et dans la base de test
vector<string> training_classes, testing_classes, testing_files;

// La matrice qui contient le vocabulaire principal
Mat vocabulary;

// Le taux d'erreur toléré pour le SVM
double c;

// Le vecteur contenant tous les SVM chargés
vector<Ptr<SVM>> svms;

// Le vecteur contenant les classes prédites
vector<string> predictions;

/**** Méthodes ****/

void getClasses();
void loadVocabulary();
void setVocabulary();
Mat calcDescriptor(string imageName);
void loadSVM();
void testSVM(string filename, Mat current_descriptor);
void computePredictResults();
double sigmoid_predict(double decision_value, double A, double B);

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

	/**** TESTING.DATA ****/

	FILE *in2;

	if ((in2 = fopen(testing_data.c_str(), "rt")) != NULL)
	{
		char buffer[100];

		while (read_line(in2, buffer, sizeof buffer))
		{
			vector<string> line;
			line = parseLine(buffer);

			testing_files.push_back(line[0]);
			testing_classes.push_back(line[1]);
		}
	}

	else
	{
		cout << "Probleme fichier : " << testing_data << endl;
		exit(-1);
	}

	fclose(in2);
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

Mat calcDescriptor(string imageName)
{
	vector<KeyPoint> imageKeypoints;
	Mat bowDescriptor;

	Mat colorImage = imread(imageName);

	if (!colorImage.cols)
	{
		cout << "Error while opening file " << imageName << endl;
	}

	else
	{
		cout << "Loading image: " << imageName << endl;
	}

	featureDetector->detect(colorImage, imageKeypoints);
	bowExtractor->compute(colorImage, imageKeypoints, bowDescriptor, 5);

	if (!bowDescriptor.cols)
	{
		cout << "Error while computing BOW histogram for file " << imageName << endl ;
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

void testSVM(string filename, Mat bowDescriptors)
{
	cout << "SVM prediction" << endl;

	string prediction;
	float best_score;
	float score, score1;

	for (size_t svm_index = 0; svm_index < svms.size(); svm_index++)
	{
		float signMul = 1.f;

		float classVal = svms[svm_index]->predict(bowDescriptors);
		float scoreVal = svms[svm_index]->predict(bowDescriptors, noArray(), StatModel::Flags::RAW_OUTPUT);

		
		// On récupère le A et B de la sigmoide associée au SVM => mise à jour personnelle du code source d'OpenCV, peut ne pas fonctionner
		// d'où la mise en commentaire
		
		//double A = svms[svm_index]->getProbA();
		//double B = svms[svm_index]->getProbB();

		signMul = (classVal < 0) == (scoreVal < 0) ? 1.f : -1.f;
		score1 = signMul * scoreVal;

		//score = sigmoid_predict(scoreVal, A, B);

		if (svm_index == 0 || score1 >= best_score)
		{
			prediction = training_classes[svm_index];
			best_score = score1;
			cout << "Best score: " << best_score << endl;
		}
	}

	cout << "Prediction: " << prediction << endl << endl;

	predictions.push_back(prediction);
}

double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value * A + B;
	if (fApB >= 0)
		return exp(-fApB) / (1.0 + exp(-fApB));
	else
		return 1.0 / (1 + exp(fApB)) ;
}

void computePredictResults()
{
	cout << endl << "********* TESTING IMAGES *********" << endl << endl;

	Mat current_descriptor;
	string image_to_open;

	loadVocabulary();
	setVocabulary();
	loadSVM();

	for (size_t file_i = 0; file_i < testing_files.size(); file_i++)
	{
		image_to_open = data_directory + TESTING_FOLDER + testing_files[file_i] + ".jpg";
		current_descriptor = calcDescriptor(image_to_open);
		testSVM(image_to_open, current_descriptor);
	}

	float global_score = 0.0;
	float round_score;

	for (size_t j = 0; j < predictions.size(); j++)
	{
		if (predictions[j] == testing_classes[j])
		{
			global_score += 1.0;
		}

		round_score = 100 * global_score / predictions.size();

		cout << "File: " << testing_files[j] << ".jpg" << endl;
		cout << "Actual class: " << testing_classes[j] << endl;
		cout << "Prediction was: " << predictions[j] << endl << endl;
	}

	cout << "GOOD PREDICTIONS: " << global_score << " / " << predictions.size() << endl;
	cout << "GLOBAL SCORE: " << trunc(round_score) << " % " << endl << endl;

	ofstream out;
	string res_file = data_directory + RESULTS_FOLDER + "testing_" + to_string((long long)nbr_cluster) + "_" + to_string((long long)c) + ".txt";
	out.open(res_file, ios::out | ios::app);
	out << "GOOD PREDICTIONS: " << global_score << " / " << predictions.size() << endl;
	out << "GLOBAL SCORE: " << trunc(round_score) << " % " << endl;
	out.close();
}

void help(char* argv[])
{
	cout << "Usage: " << argv[0] << " Data_folder Nbr_cluster C" << endl;
}

int main(int argc, char* argv[])
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	if (argc != 4)
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

	training_data = data_directory + TRAINING_DATA_FILE;
	testing_data = data_directory + TESTING_DATA_FILE;

	getClasses();
	computePredictResults();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

	cout << "TESTING TIME: " << convertTime(duration) << endl;

	ofstream out;
	string res_file = data_directory + RESULTS_FOLDER + "testing_" + to_string((long long)nbr_cluster) + "_" + to_string((long long)c) + ".txt";
	out.open(res_file, ios::out | ios::app);
	out << "TESTING TIME: " << convertTime(duration);
	out.close();

	return 0;
}
