#include "Soft.cpp"

using namespace std;
using namespace cv;

int nbr_cluster;

Mat loadBowDescriptor(string filename, string type) {

	cout << "Je suis dans le loadBowDescriptor" << endl;
	Mat bowDescriptor;
	FileStorage fs(  filename, FileStorage::READ );
	if ( fs.isOpened() )  {
		cout << "Fichier ouvert" << endl;
		if ( type == "imageDescriptor" )
			fs["imageDescriptor"] >> bowDescriptor;
		if ( type == "vocabulary" )
			fs["vocabulary"] >> bowDescriptor;
	}
	else {
		cout << "Erreur lors de l'ouverture du fichier" << filename << endl;
		exit(-1);
	}

	return bowDescriptor;

}

Mat calcDescriptor(Mat vocabulary, string imageName) {

	cout << "Calcul du descripteur" << endl;
	stringstream ss;
	ss.str("");
	ss << imageName << "." << nbr_cluster;
	vector<KeyPoint> imageKeypoints;
	Mat bowDescriptor;
	bowExtractor->setVocabulary( vocabulary );
	Mat colorImage = imread(imageName);
	if (!colorImage.cols) {
		cout << "erreur lors de l'ouverture du fichier " << endl;
	}
	featureDetector->detect ( colorImage, imageKeypoints );
	bowExtractor->compute( colorImage, imageKeypoints, bowDescriptor, 5);
	if (!bowDescriptor.cols) {

		cout << "bowDescriptor = " << bowDescriptor << endl;
	}
	else {
		cout << "yes" << endl;
	}
	writeBowImageDescriptor(ss.str(), bowDescriptor, "imageDescriptor");
	return bowDescriptor;
}

void travelFile(string directory, vector<int> nbr_clusters)
{

	string trainDirectory = directory;
	FILE *in;
	stringstream ss;
	string filephoto = directory + "/" + "photo.data";
	vector<string> classes;
	Mat bowDescriptors;
	vector<vector<string>> lines;
// recherche de classe
	if ((in = fopen(filephoto.c_str(), "rt")) != NULL) {

		char buffer[100];
		while (read_line(in, buffer, sizeof buffer)) {
			vector<string> line;
			line = parseLine(buffer);
			lines.push_back(line);


			if (find(classes.begin(), classes.end(), line[1]) == classes.end()) {
				classes.push_back(line[1]);
			}
		}
	}

	fclose(in);

	for (int cln = 0; cln < nbr_clusters.size(); cln++) {

		ss.str("");
		nbr_cluster = nbr_clusters[cln]; // -> TODO
		vector<string> line = lines[0];
		string filename;
		ss << directory << "vocabulary." << nbr_cluster << ".xml.gz";
		// charger le vocabulaire
		Mat temp = loadBowDescriptor(ss.str(), "vocabulary");

		Mat* trainData = NULL;
		trainData = new Mat( (int)lines.size(), temp.rows, CV_32FC1 );
		cout << "Calcul des descripteurs: nombre cluster = " << nbr_cluster << " ..." << endl;
		for (int i = 0; i < lines.size(); i++) {
			ss.str("");
			ss << trainDirectory << "/" + lines[i][0] << ".jpg";
			cout << "line = " << ss.str() << " " << nbr_cluster << endl;
			// calculer le descripteur
			Mat bowDescriptor = calcDescriptor(temp, ss.str());
		}

	}
}

int main(int argc, char* argv[]) {

	cv::initModule_nonfree();
	featureDetector = FeatureDetector::create(argv[1]);
	descExtractor = DescriptorExtractor::create(argv[2]);
	if (featureDetector.empty() || descExtractor.empty()) {
		cout << "Error in detector and extractor";
		exit(-1);
	}
	descMatcher = DescriptorMatcher::create(argv[3]);
	if (descMatcher.empty()) {
		cout << "probleme matcher";
	}

	cout << argv[1] << 	argv[2] << argv[3] << argv[4] << argv[5] << "soft" << endl;
	bowExtractor = new SoftBOWImgDescriptorExtractor( descExtractor, descMatcher );
//vector<int> nbr_cluster {100,200,500,1000,2000,4000};
	vector<int> nbr_cluster;
	nbr_cluster.push_back(atoi(argv[4]));
	travelFile(argv[5], nbr_cluster);
	cout << "Finish" << endl;
}
