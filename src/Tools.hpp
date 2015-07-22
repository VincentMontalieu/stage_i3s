#ifndef TOOLS_HPP
#define TOOLS_HPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"

#include <vector>
#include <string>


/***** Chemins vers des dossiers ou fichiers de l'architecture de données *****/

const std::string MAIN_VOCAB_FOLDER = "main_vocab/";
const std::string PLANTS_SUMMARY_FOLDER = "plants_summary/";
const std::string PLANTS_SVM_FOLDER = "plants_svm/";
const std::string PLANTS_VOCABS_FOLDER = "plants_vocabs/";
const std::string RESULTS_FOLDER = "results/";
const std::string TESTING_FOLDER = "testing/";
const std::string TRAINING_FOLDER = "training/";
const std::string TRAINING_DATA_FILE = "plants_summary/training.data";
const std::string TESTING_DATA_FILE = "plants_summary/testing.data";


/***** Déclaration des méthodes utiles à plusieurs modules *****/

int read_line(FILE *in, char *buffer, size_t max);
std::vector<std::string> parseLine(char line[100]);
void setDataDirectoryPath(std::string& path);
void writeBOWImageDescriptor(const std::string& file, const cv::Mat& bowImageDescriptor, std::string name);
cv::Mat loadBOWDescriptor(std::string filename, std::string type);
std::string convertTime(int seconds);

#endif