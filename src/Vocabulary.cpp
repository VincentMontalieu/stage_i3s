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

/**** Méthodes ****/

void calcDescriptor();
void createMainVocabulary();
void createBOWHistograms();

void createMainVocabulary()
{
	cout << endl << endl << "********* MAIN VOCABULARY *********" << endl << endl;

	FILE *in;
	int y = 0;

	vector<Mat> descriptors;
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;

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

void help(char* argv[])
{
	cout << "Usage: " << argv[0] << " Data_folder Nbr_cluster" << endl;
}

int main(int argc, char* argv[])
{	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	if (argc != 3)
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

	training_data = data_directory + TRAINING_DATA_FILE;

	createMainVocabulary();
	createBOWHistograms();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

	cout << "Clustering time: " << convertTime(duration) << endl;

	ofstream out;
	string res_file = data_directory + RESULTS_FOLDER + "training_vocabulary_" + to_string((long long)nbr_cluster) + ".txt";
	remove(res_file.c_str());
	out.open(res_file, ios::out | ios::app);
	out << "Clustering time: " << convertTime(duration);
	out.close();

	return 0;
}
