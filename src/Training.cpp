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

// La matrice qui contient le vocabulaire principal
Mat vocabulary;

// Le taux d'erreur toléré pour le SVM
int c;

/**** Méthodes ****/
void calcDescriptor(size_t i);
void createMainVocabulary();
void createBOWHistograms();
void trainSVM();

void createMainVocabulary()
{
	cout << endl << endl << "********* MAIN VOCABULARY *********" << endl << endl;

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
	cout << endl << endl << "********* HISTOGRAMS *********" << endl << endl;
	cout << "CREATING HISTOGRAMS NOW ...." << endl;
	cout << nbr_cluster.size() << " BOW trainer(s) for this session" << endl << endl;

	for (size_t i = 0; i < nbr_cluster.size(); i++)
	{
		cout << "Using " << nbr_cluster[i] << " ...." << endl << endl;
		string vocabulary_file_path = data_directory + MAIN_VOCAB_FOLDER + "vocabulary." + to_string(nbr_cluster[i]) + ".xml.gz";
		vocabulary = loadBOWDescriptor(vocabulary_file_path, "vocabulary");
		cout << "Vocabulary loaded" << endl;
		calcDescriptor(i);
	}

	cout << "HISTOGRAMS CREATED" << endl;
}

void calcDescriptor(size_t cln)
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

		writeBOWImageDescriptor(data_directory + PLANTS_VOCABS_FOLDER + single_img_name + "." + to_string(nbr_cluster[cln]), bowDescriptor, "imageDescriptor");
	}
}

void trainSVM()
{
	cout << endl << endl << "********* SVM TRAINING *********" << endl << endl;

	for (size_t cln = 0; cln < nbr_cluster.size(); cln++)
	{
		Mat* trainData = NULL;
		trainData = new Mat((int)text_lines.size(), vocabulary.rows, CV_32FC1);

		cout << "Loading descriptors .... " << endl;

		for (size_t i = 0; i < text_lines.size(); i++)
		{
			string descriptor_file_path = data_directory + PLANTS_VOCABS_FOLDER + text_lines[i][0] + ".jpg." + to_string(nbr_cluster[cln]) + ".xml.gz";
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

			CvSVMParams params;
			params.svm_type = CvSVM::C_SVC;
			params.kernel_type = CvSVM::LINEAR;
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 10); // Les deux derniers paramètres sont le nombre max d'itérations et la précision
			params.C = c;

			CvSVM* svm = new CvSVM();

			cout << "SVM training on class: " << classes[i] << " ...." << endl;

			// Train the SVM
			svm->train(*trainData, *responses, Mat(), Mat(), params);

			cout << "SVM trained for " << classes[i] << endl;
			string svm_file_to_save = data_directory + PLANTS_SVM_FOLDER + "svm:" + classes[i] + "." + to_string(nbr_cluster[cln]) + "." + to_string(c) + ".xml.gz";
			cout << "Saving SVM training file in " << svm_file_to_save << endl << endl;

			svm->save(svm_file_to_save.c_str());

			delete svm;
			delete responses;
		}

		delete trainData;
	}

	cout << endl << "DONE TRAINING SVM" << endl;
}

void help(char* argv[])
{
	cout << "Usage: " << argv[0] << "FeatureDetector DescriptorExtractor DescriptorMatcher NbrCluster SVM_c Directory" << endl;
}

int main(int argc, char* argv[])
{
	initModule_nonfree();

	if (argc < 7)
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

	for (int i = 4; i < argc - 2; i++)
	{
		nbr_cluster.push_back(atoi(argv[i]));
	}

	c = atoi(argv[5]);

	data_directory = argv[6];

	setDataDirectoryPath(data_directory);
	training_data = data_directory + TRAINING_DATA_FILE;

	createMainVocabulary();
	createBOWHistograms();
	trainSVM();

	return 0;
}