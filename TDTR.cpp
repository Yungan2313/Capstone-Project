#include<iostream>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
#include<math.h>
#include "sys/time.h"
#include<algorithm>
using namespace std;
typedef struct Point{
	double lat;
	double lon;
	double time;
}Point;
vector<Point> points;
void gpsreader(string filename){
	ifstream fin(filename.c_str());
	if (!fin){
		cout << "open file error !" << endl;
		exit(0);
	}
	while (fin) {
        string line;
        getline(fin, line);

        if (line.empty()) {  // 如果遇到空行，跳過
            continue;
        }

        stringstream sin(line);
        string temp;

        Point point;

        // 透過逗號來分割資料
        getline(sin, temp, ',');
        point.lat = stod(temp);  // 將字串轉換成 double

        getline(sin, temp, ',');
        point.lon = stod(temp);

        getline(sin, temp, ',');
        point.time = stod(temp);

        points.push_back(point);

        // cout << point.lat << " " << point.lon << " " << point.time << endl;
    
	}
	points.pop_back();
	return;
}
double cacl_SED(Point s, Point m, Point e){
    double numerator = m.time - s.time;
    double denominator = e.time - s.time;
    double time_ratio = denominator == 0 ? 1 : numerator / denominator;
    double lat = s.lat + (e.lat - s.lat)*time_ratio;
    double lon = s.lon + (e.lon - s.lon)*time_ratio;
    double lat_diff = lat - m.lat;
    double lon_diff = lon - m.lon;
    return sqrt(lat_diff*lat_diff + lon_diff*lon_diff);
  }
vector<int> DouglasPeucker(int start, int last, double epsilon){

	double dmax = 0;
	int index = start;
	vector<int> recResults;
	for (int i = start + 1; i < last; i++){
		double d = cacl_SED(points[start], points[i], points[last]);
		if (d > dmax){
			index = i;
			dmax = d;
		}
	}
	if (dmax > epsilon){
			vector<int> recResults1 = DouglasPeucker(start, index, epsilon);
			vector<int> recResults2 = DouglasPeucker(index, last, epsilon);
			recResults.insert(recResults.end(), recResults1.begin(), recResults1.end());
			recResults.insert(recResults.end(), recResults2.begin(), recResults2.end());
	}
	else{
			recResults.push_back(start);
			recResults.push_back(last);
		}
	

	return recResults;

}

int main(int argc, char** args){
    string filename = args[1];
    double epsilon = atof(args[2]);
    string save_filename = args[3];
    gpsreader(filename);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    vector<int> cmp_index = DouglasPeucker(0, points.size() - 1, epsilon);
    gettimeofday(&end, NULL);
    long timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    sort(cmp_index.begin(), cmp_index.end());
    cmp_index.erase(unique(cmp_index.begin(), cmp_index.end()), cmp_index.end());

    if (save_filename == "-") {
        // 輸出到 stdout
        for (int i = 0; i < cmp_index.size(); i++) {
            printf("%.6lf,%.6lf,%.0lf\n", points[cmp_index[i]].lat, points[cmp_index[i]].lon, points[cmp_index[i]].time);
        }
    } else {
        // 輸出到檔案
        FILE* s_fp = fopen(save_filename.c_str(), "w+");
        for (int i = 0; i < cmp_index.size(); i++) {
            fprintf(s_fp, "%.6lf,%.6lf,%.0lf\n", points[cmp_index[i]].lat, points[cmp_index[i]].lon, points[cmp_index[i]].time);
        }
        fprintf(s_fp, "%.6lf\n", timeuse / 1000000.0);
        fclose(s_fp);
    }

    return 0;
}

