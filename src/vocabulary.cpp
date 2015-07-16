#include "tools.hpp"

using namespace std;
using namespace cv;

/***** Outils de détection et d'extraction de features *****/

Ptr<FeatureDetector> featureDetector;
Ptr<DescriptorExtractor> descExtractor;
Ptr<BOWImgDescriptorExtractor> bowExtractor;
Ptr<DescriptorMatcher> descMatcher;


// Vecteur contenant le ou les nombres de clusters à utiliser
vector<int> nbr_cluster;

// Le chemin vers le dossier principal contenant les données
string data_directory;

class Photo
{
public:
	string filename;
	Mat bowDescriptors;
	vector<KeyPoint> imageKeypoints;
	Mat imageDescriptors;

	Photo(string filenamep, Mat bowDescriptorsp, vector<KeyPoint> imageKeypointsp, Mat imageDescriptorsp)
	{
		filename = filenamep;
		bowDescriptors = bowDescriptorsp;
		imageKeypoints = imageKeypointsp;
		imageDescriptors = imageDescriptorsp;
	}
};

void createMainVocabulary()
{
	vector<Photo> photos;
	FILE *in;
	vector<string> fileList;
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

	string file_to_open = data_directory + TRAINING_DATA_FILE;

	if ((in = fopen(file_to_open.c_str(), "rt")) != NULL)
	{
		char buffer[100];

		// pour chaque ligne de training.data
		while (read_line(in, buffer, sizeof buffer))
		{
			cout << "Parsing line " << y << " from " << TRAINING_DATA_FILE << endl;
			cout << "CONTENT: " << buffer ;
			vector<string> line;

			// return element de chaque ligne du fichier
			line = parseLine(buffer);

			string file = data_directory + TRAINING_FOLDER + line[0];
			string imgfile = file + ".jpg";

			Mat colorImage = imread(imgfile.c_str());
			// compute key point
			vector<KeyPoint> imageKeypoints;

			featureDetector->detect ( colorImage, imageKeypoints );
			// compute descriptor
			Mat imageDescriptors;
			descExtractor->compute ( colorImage, imageKeypoints, imageDescriptors);
			int descCount = imageDescriptors.rows;
			cout << "Matching image found: " << imgfile << endl;
			cout << "Features detected: " << descCount << endl << endl;

			if (!imageDescriptors.empty())
			{
				fileList.push_back(file);

				for (int j = 0; j < descCount; j++)
				{
					for (size_t i = 0; i < bowTrainerList.size(); i++)
					{
						// preparation pour le clustring
						bowTrainerList[i]->add(imageDescriptors.row(j));
					}
				}

				photos.push_back(Photo(file, Mat(), imageKeypoints, imageDescriptors));

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
		string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string(nbr_cluster[i]);
		cout << "Clustering completed" << endl << endl;
		writeBowImageDescriptor(vocabulary_file_path, vocabulary, "vocabulary");
		vocabulary_file_path.clear();
		vocabulary.release();
	}
}

// void createBOWHistograms(argv[5], nbr_cluster)
// {

// }

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
	bowExtractor = new BOWImgDescriptorExtractor(descExtractor, descMatcher); // HARD ASSIGNMENT
	// bowExtractor = new SoftBOWImgDescriptorExtractor(descExtractor, descMatcher); // SOFT ASSIGNMENT

	for (int i = 4; i < argc - 1; i++)
	{
		nbr_cluster.push_back(atoi(argv[i]));
	}

	data_directory = argv[5];

	setDataDirectoryPath(data_directory);

	createMainVocabulary();
	//createBOWHistograms(argv[5], nbr_cluster);

	return 0;
}