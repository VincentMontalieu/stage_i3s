#include "Tools.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

/***** Implémentation des méthodes *****/

int read_line(FILE *in, char *buffer, size_t max)
{
	return fgets(buffer, max, in) == buffer;
}

vector<string> parseLine(char line[100])
{
	vector<string> res;
	int i = 0;
	string temp = "";
	while ( line[i] != '\n') {
		if (line[i] == ':') {
			res.push_back(temp);
			temp = "";
		}
		else {
			temp = temp + line[i];
		}
		i++;
	}
	res.push_back(temp);
	return res;
}

void setDataDirectoryPath(string& path)
{
	if (!path.empty())
	{
		if (path.back() != '/')
		{
			path += "/";
		}
	}
}

void writeBOWImageDescriptor(const string& file, const Mat& bowImageDescriptor, string name)
{
	cout << "Saving in " << file << ".xml.gz" << endl << endl;
	FileStorage fs(file + ".xml.gz", FileStorage::WRITE );

	if (fs.isOpened())
	{
		fs << name << bowImageDescriptor;
	}
}

Mat loadBOWDescriptor(string filename, string type)
{
	Mat bowDescriptor;
	FileStorage fs(filename, FileStorage::READ);

	if (fs.isOpened())
	{
		if ( type == "imageDescriptor" )
		{
			fs["imageDescriptor"] >> bowDescriptor;
		}

		else if ( type == "vocabulary" )
		{
			fs["vocabulary"] >> bowDescriptor;
		}
	}

	else
	{
		cout << "Couldn't open " << filename << endl;
		exit(-1);
	}

	return bowDescriptor;
}

string convertTime(int seconds)
{
	int minutes = seconds / 60;
	int hours = minutes / 60;
	string time_string = to_string(hours) + " hours " + to_string(minutes%60) + " min " + to_string(seconds%60) + " sec";
	return time_string;
}