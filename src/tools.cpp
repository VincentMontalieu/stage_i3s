#include "tools.hpp"

using namespace std;
using namespace cv;

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

void writeBowImageDescriptor(const string& file, const Mat& bowImageDescriptor, string name)
{
	cout << "Saving in " << file << ".xml.gz" << endl;
	FileStorage fs(file + ".xml.gz", FileStorage::WRITE );

	if (fs.isOpened())
	{
		fs << name << bowImageDescriptor;
		cout << "DONE" << endl;
	}
}