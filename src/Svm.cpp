#include "ImageData.hpp"
#include "Soft.hpp"
#include <chrono>
#include <fstream>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace cv::xfeatures2d;
using namespace cv::ml;

// le nombre de clusters à utiliser
int nbr_cluster;

// Le chemin vers le dossier principal contenant les données ainsi que celui vers le fichier training.data
string data_directory, training_data;

// Représente une ligne de texte;
vector<string> text_line;

// Représente un ensemble de lignes de texte
vector<vector<string>> text_lines;

// Représente l'ensemble des classes (plantes) présentes dans la base d'apprentisage
vector<string> classes;

// La matrice qui contient le vocabulaire principal
Mat vocabulary;

// Le taux d'erreur toléré pour le SVM
double c;

/**** Méthodes ****/

void getClasses();
void loadVocabulary();
void trainSVM();

void getClasses()
{
	/**** TRAINING.DATA ****/

	FILE *in;

	if ((in = fopen(training_data.c_str(), "rt")) != NULL)
	{
		char buffer[100];

		while (read_line(in, buffer, sizeof buffer))
		{
			text_line = parseLine(buffer);
			text_lines.push_back(text_line);

			if (find(classes.begin(), classes.end(), text_line[1]) == classes.end())
			{
				classes.push_back(text_line[1]);
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
	string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string((long long)nbr_cluster) + ".xml.gz";
	vocabulary = loadBOWDescriptor(vocabulary_file_path, "vocabulary");
	cout << "Vocabulary loaded" << endl;
}

void trainSVM()
{
	cout << endl << endl << "********* SVM TRAINING *********" << endl << endl;

	Mat* trainData = NULL;
	trainData = new Mat((int)text_lines.size(), vocabulary.rows, CV_32FC1);

	cout << "Loading descriptors .... " << endl;

	for (size_t i = 0; i < text_lines.size(); i++)
	{
		string descriptor_file_path = data_directory + PLANTS_VOCABS_FOLDER + text_lines[i][0] + ".jpg." + to_string((long long)nbr_cluster) + ".xml.gz";
		Mat bowDescriptor = loadBOWDescriptor(descriptor_file_path, "imageDescriptor");
		Mat submat = trainData->row((int)i);
		bowDescriptor.copyTo(submat);
	}

	float w1 = 0, w2 = 0;

	for (size_t i = 0; i < classes.size(); i++)
	{
		// Matrice contenant les labels, ici 1 ou -1
		Mat* responses = new Mat((int)text_lines.size(), 1, CV_32SC1);

		for (size_t j = 0; j < text_lines.size(); j++)
		{
			responses->at<int>((int)j) = (text_lines[j][1] == classes[i]) ? 1 : -1;
			(text_lines[j][1] == classes[i]) ? w1++ : w2++;

		}

		Mat weights = Mat::zeros(2, 1, CV_32FC1);
		weights.at<float>(0) = (float) w1 / (w1 + w2);
		weights.at<float>(1) = (float) w2 / (w1 + w2);

		Ptr<SVM> svm = SVM::create();
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::LINEAR);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
		svm->setC(c);
		svm->setClassWeights(weights);

		cout << "SVM training on class: " << classes[i] << " ...." << endl;

		// Train the SVM
		//svm->trainAuto(trainData, 10, SVM::getDefaultGrid(SVM::C), SVM::getDefaultGrid(SVM::GAMMA),  SVM::getDefaultGrid(SVM::P),  SVM::getDefaultGrid(SVM::NU),  SVM::getDefaultGrid(SVM::COEF),  SVM::getDefaultGrid(SVM::DEGREE), true);
		Ptr<TrainData> data = TrainData::create( *trainData, ROW_SAMPLE, *responses );
		svm->train_probability(data);

		cout << "SVM trained for " << classes[i] << endl;
		string svm_file_to_save = data_directory + PLANTS_SVM_FOLDER + "svm:" + classes[i] + "." + to_string((long long)nbr_cluster) + "." + to_string((long long)c) + ".xml.gz";
		cout << "Saving SVM training file in " << svm_file_to_save << endl << endl;

		svm->save(svm_file_to_save);

		delete responses;
	}

	delete trainData;

	cout << endl << "DONE TRAINING SVM" << endl;
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

	data_directory = argv[1];
	nbr_cluster = atoi(argv[2]);
	c = atof(argv[3]);

	training_data = data_directory + TRAINING_DATA_FILE;

	getClasses();
	loadVocabulary();
	trainSVM();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

	cout << "SVM training time: " << convertTime(duration) << endl;

	ofstream out;
	string res_file = data_directory + RESULTS_FOLDER + "training_SVM_" + to_string((long long)nbr_cluster) + "_" + to_string((long long)c) + ".txt";;
	remove(res_file.c_str());
	out.open(res_file, ios::out | ios::app);
	out << "SVM training time: " << convertTime(duration);
	out.close();

	return 0;
}
