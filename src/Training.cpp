#include "Tools.hpp"
#include "ImageData.hpp"
#include "Soft.hpp"

using namespace std;
using namespace cv;

/***** Outils de détection et d'extraction de features *****/

Ptr<FeatureDetector> featureDetector;
Ptr<DescriptorExtractor> descExtractor;
Ptr<SoftBOWImgDescriptorExtractor> bowExtractor;
Ptr<DescriptorMatcher> descMatcher;


// Vecteur contenant le ou les nombres de clusters à utiliser
vector<int> nbr_cluster;

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

/**** Méthodes ****/
void calcDescriptor(Mat vocabulary, vector<ImageData> data);
void createMainVocabulary();
void createBOWHistograms();
void trainSVM();

void createMainVocabulary()
{
	FILE *in;
	int y = 0;

	vector<Mat> descriptors;

	vector<BOWKMeansTrainer*> bowTrainerList;
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;

	// generation des bowTrainer pour les differents cluster
	for (size_t i = 0; i < nbr_cluster.size(); i++)
	{
		BOWKMeansTrainer* bowTrainer = new BOWKMeansTrainer(nbr_cluster[i], terminate_criterion, 3, KMEANS_PP_CENTERS);
		bowTrainerList.push_back(bowTrainer);
	}

	int desNull = 0;

	if ((in = fopen(training_data.c_str(), "rt")) != NULL)
	{
		char buffer[100];

		// Pour chaque ligne de training.data
		while (read_line(in, buffer, sizeof buffer))
		{
			cout << "Parsing text_line " << y << " from " << TRAINING_DATA_FILE << endl;
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
					for (size_t i = 0; i < bowTrainerList.size(); i++)
					{
						// preparation pour le clustring
						bowTrainerList[i]->add(imageDescriptors.row(j));
					}
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
		cout << bowTrainerList.size() << " BOW trainer(s) for this session" << endl << endl;

		fclose(in);
	}

	else
	{
		cout << "Probleme fichier : " << data_directory << TRAINING_DATA_FILE << endl;
	}

	for (size_t i = 0; i < bowTrainerList.size(); i++)
	{
		cout << "Clustering .... " << nbr_cluster[i] << endl;
		Mat vocabulary = bowTrainerList[i]->cluster();
		cout << "Clustering completed" << endl << endl;

		string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string(nbr_cluster[i]);
		writeBOWImageDescriptor(vocabulary_file_path, vocabulary, "vocabulary");
		cout << "DONE" << endl;

		vocabulary_file_path.clear();
		vocabulary.release();
	}
}

void createBOWHistograms()
{
	cout << "CREATING HISTOGRAMS NOW ...." << endl;
	cout << nbr_cluster.size() << " BOW trainer(s) for this session" << endl << endl;

	for (size_t i = 0; i < nbr_cluster.size(); i++)
	{
		cout << "Using " << nbr_cluster[i] << " ...." << endl << endl;
		string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string(nbr_cluster[i]) + ".xml.gz";
		Mat vocabulary = loadBOWDescriptor(vocabulary_file_path, "vocabulary");
		cout << "Vocabulary loaded" << endl;
		calcDescriptor(vocabulary, plants_pics_data);
	}

	cout << "HISTOGRAMS CREATED" << endl;
}

void calcDescriptor(Mat vocabulary, vector<ImageData> data)
{
	bowExtractor->setVocabulary(vocabulary);
	cout << "Vocabulary set" << endl;

	for (size_t i = 0; i < data.size(); i++)
	{
		Mat bowDescriptor;
		string single_img_name = data[i].filename;
		string img_to_open = data_directory + TRAINING_FOLDER + single_img_name;
		vector<KeyPoint> features = data[i].imageKeypoints;
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

		writeBOWImageDescriptor(data_directory + PLANTS_VOCABS_FOLDER + single_img_name, bowDescriptor, "imageDescriptor");
	}
}

void trainSVM()
{

}

void help(char* argv[])
{
	cout << "Usage: " << argv[0] << "FeatureDetector DescriptorExtractor DescriptorMatcher NbrCluster Directory" << endl;
}

int main(int argc, char* argv[])
{
	initModule_nonfree();

	if (argc < 6)
	{
		help(argv);
		exit(-1);
	}

	// detecteur de POI
	featureDetector = FeatureDetector::create(argv[1]);

	// descripteur de POI
	descExtractor = DescriptorExtractor::create(argv[2]);

	if (featureDetector.empty() || descExtractor.empty())
	{
		cout << "Error in detector and extractor";
		exit(-1);
	}

	descMatcher = DescriptorMatcher::create(argv[3]);

	if (descMatcher.empty())
	{
		cout << "probleme matcher";
	}

	/*** Generateur de BOW ***/
	bowExtractor = new SoftBOWImgDescriptorExtractor(descExtractor, descMatcher); // SOFT ASSIGNMENT

	for (int i = 4; i < argc - 1; i++)
	{
		nbr_cluster.push_back(atoi(argv[i]));
	}

	data_directory = argv[5];

	setDataDirectoryPath(data_directory);
	training_data = data_directory + TRAINING_DATA_FILE;

	createMainVocabulary();
	createBOWHistograms();
	trainSVM();

	return 0;
}