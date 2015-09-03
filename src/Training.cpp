#include "ImageData.hpp"
#include "Soft.hpp"
#include <chrono>
#include <fstream>

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


// le nombre de clusters à utiliser
int nbr_cluster;

// Le chemin vers le dossier principal contenant les données ainsi que celui vers le fichier training.data
string data_directory, training_data;

// Vecteur contenant un objet ImageData par photo de plante analysée
vector<ImageData> plants_pics_data;

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

// Le myGamma du Kernel RBF
double myGamma;

/**** Méthodes ****/

void calcDescriptor();
void createMainVocabulary();
void createBOWHistograms();
void trainSVM();


void createMainVocabulary()
{
	cout << endl << endl << "********* MAIN VOCABULARY *********" << endl << endl;

	FILE *in;
	int y = 0;

	vector<Mat> descriptors;
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;

	// generation des bowTrainer pour les differents cluster
	BOWKMeansTrainer* bowTrainer = new BOWKMeansTrainer(nbr_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);

	int desNull = 0;

	if ((in = fopen(training_data.c_str(), "rt")) != NULL)
	{
		char buffer[100];

		// Pour chaque ligne de training.data
		while (read_line(in, buffer, sizeof buffer))
		{
			cout << "Parsing line " << y << " from " << TRAINING_DATA_FILE << endl;
			cout << "CONTENT: " << buffer ;

			text_line = parseLine(buffer);
			text_lines.push_back(text_line);

			if (find(classes.begin(), classes.end(), text_line[1]) == classes.end())
			{
				classes.push_back(text_line[1]);
			}

			string imgfile = data_directory + TRAINING_FOLDER + text_line[0] + ".jpg";

			Mat colorImage = imread(imgfile.c_str());

			// compute key point
			vector<KeyPoint> imageKeypoints;
			featureDetector->detect(colorImage, imageKeypoints);

			// compute descriptor
			Mat imageDescriptors;
			descExtractor->compute(colorImage, imageKeypoints, imageDescriptors);

			int descCount = imageDescriptors.rows;

			cout << "Matching image found: " << imgfile << endl;
			cout << "Features detected: " << descCount << endl << endl;

			if (!imageDescriptors.empty())
			{
				for (int j = 0; j < descCount; j++)
				{
					// preparation pour le clustring
					bowTrainer->add(imageDescriptors.row(j));
				}

				plants_pics_data.push_back(ImageData(text_line[0] + ".jpg" , Mat(), imageKeypoints, imageDescriptors));

			}

			else
			{
				desNull++;
			}

			y++;
		}

		cout << "Images without any features: " << desNull << endl ;
		cout << "1 BOW trainer for this session" << endl << endl;

		fclose(in);
	}

	else
	{
		cout << "Probleme fichier : " << data_directory << TRAINING_DATA_FILE << endl;
	}

	cout << "Clustering .... " << nbr_cluster << endl;
	Mat vocabulary = bowTrainer->cluster();
	cout << "Clustering completed" << endl << endl;

	string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string((long long)nbr_cluster);
	writeBOWImageDescriptor(vocabulary_file_path, vocabulary, "vocabulary");
	cout << "DONE" << endl;

	vocabulary_file_path.clear();
	vocabulary.release();
}

void createBOWHistograms()
{
	cout << endl << endl << "********* HISTOGRAMS *********" << endl << endl;
	cout << "CREATING HISTOGRAMS NOW ...." << endl;
	cout << "1 BOW trainer for this session" << endl << endl;

	cout << "Using " << nbr_cluster << " ...." << endl << endl;
	string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string((long long)nbr_cluster) + ".xml.gz";
	vocabulary = loadBOWDescriptor(vocabulary_file_path, "vocabulary");
	cout << "Vocabulary loaded" << endl;
	calcDescriptor();
	cout << "HISTOGRAMS CREATED" << endl;
}

void calcDescriptor()
{
	bowExtractor->setVocabulary(vocabulary);
	cout << "Vocabulary set" << endl;

	for (size_t i = 0; i < plants_pics_data.size(); i++)
	{
		Mat bowDescriptor;
		string single_img_name = plants_pics_data[i].filename;
		string img_to_open = data_directory + TRAINING_FOLDER + single_img_name;
		vector<KeyPoint> features = plants_pics_data[i].imageKeypoints;
		Mat colorImage = imread(img_to_open);

		cout << img_to_open << " loaded" << endl;

		if (!colorImage.cols)
		{
			cout << "Error while opening file " << img_to_open << endl;
		}

		// bowExtractor->compute(colorImage, features, bowDescriptor); // HARD ASSIGNMENT

		// Le dernier paramètre est le nombre de plus proches voisins autorisés pour le soft assignment
		bowExtractor->compute(colorImage, features, bowDescriptor, 5); // SOFT ASSIGNMENT

		cout << "Histogram computed for " << img_to_open << endl;

		if (!bowDescriptor.cols)
		{
			cout << "Error while computing BOW histogram for file " << img_to_open << endl;
		}

		writeBOWImageDescriptor(data_directory + PLANTS_VOCABS_FOLDER + single_img_name + "." + to_string((long long)nbr_cluster), bowDescriptor, "imageDescriptor");
	}
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

	for (size_t i = 0; i < classes.size(); i++)
	{
		// Matrice contenant les labels, ici 1 ou -1
		Mat* responses = new Mat((int)text_lines.size(), 1, CV_32SC1);

		for (size_t j = 0; j < text_lines.size(); j++)
		{
			responses->at<int>((int)j) = (text_lines[j][1] == classes[i]) ? 1 : -1;
		}

		Ptr<SVM> svm = SVM::create();
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::RBF);
		svm->setGamma(myGamma);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
		svm->setC(c);

		cout << "SVM training on class: " << classes[i] << " ...." << endl;

		// Train the SVM
		svm->train(*trainData, ROW_SAMPLE, *responses);

		cout << "SVM trained for " << classes[i] << endl;
		string svm_file_to_save = data_directory + PLANTS_SVM_FOLDER + "svm:" + classes[i] + "." + to_string((long long)nbr_cluster) + "." + to_string((long long)c) + "." + to_string((long long)myGamma) + ".xml.gz";
		cout << "Saving SVM training file in " << svm_file_to_save << endl << endl;

		svm->save(svm_file_to_save);

		delete responses;
	}

	delete trainData;

	cout << endl << "DONE TRAINING SVM" << endl;
}

void help(char* argv[])
{
	cout << "Usage: " << argv[0] << " Data_folder Nbr_cluster C gamma" << endl;
}

int main(int argc, char* argv[])
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	if (argc != 5)
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
	myGamma = atof(argv[4]);

	training_data = data_directory + TRAINING_DATA_FILE;

	createMainVocabulary();
	createBOWHistograms();
	trainSVM();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

	cout << "TRAINING TIME: " << convertTime(duration) << endl;

	ofstream out;
	string res_file = data_directory + RESULTS_FOLDER + "training_" + to_string((long long)nbr_cluster) + "_" + to_string((long long)c) + "_" + to_string((long long)myGamma) + ".txt";;
	out.open(res_file, ios::out | ios::app);
	out << "TRAINING TIME: " << convertTime(duration);
	out.close();

	return 0;
}
