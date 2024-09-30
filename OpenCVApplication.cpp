// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

using namespace std;

#define IMG_SIZE 20		 // for image letter size, dimensions IMG_SIZE x IMG_SIZE
#define ZONE_SIZE 4      // for zoning, dimension of a zone is ZONE_SIZE x ZONE_SIZE, has to be a divisor of IMG_SIZE
#define NUM_ZONES (IMG_SIZE / ZONE_SIZE) * (IMG_SIZE / ZONE_SIZE)  // number of zones
#define NRCLASSES 128    // ASCII code contains 128 characters

int numberofPixels(Mat img)
{
	int n = 0;
	for (int i = 0;i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
				n++;
		}
	}
	return n;
}

Mat binarization(Mat img, int threshold) 
{
	Mat res(img.rows, img.cols, CV_8UC1);

	for (int i = 0;i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) < threshold)
				res.at<uchar>(i, j) = 0;
			else
				res.at<uchar>(i, j) = 255;
		}

	return res;
}

void horizontal_hist(Mat img, int* histH)
{
	for (int i = 0; i < img.rows; i++)
	{
		int black_pixels = 0;
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
				black_pixels++;
		}

		histH[i] = black_pixels;
	}
}

void vertical_hist(Mat img, int* histV)
{
	for (int j = 0; j < img.cols; j++)
	{
		int black_pixels = 0;
		for (int i = 0; i < img.rows; i++)
		{
			if (img.at<uchar>(i, j) == 0)
				black_pixels++;
		}
		histV[j] = black_pixels;
	}
}

void zoning(Mat img, int* zones)
{
	int z = 0;
	for (int r = ZONE_SIZE; r <= img.rows; r += ZONE_SIZE)
	{
		for (int c = ZONE_SIZE; c <= img.cols; c += ZONE_SIZE)
		{
			int n = 0;
			for (int i = r - ZONE_SIZE; i < r; i++)
			{
				for (int j = c - ZONE_SIZE; j < c; j++)
					if (img.at<uchar>(i, j) == 0)
						n++;
			}
			zones[z] = n;
			z++;
		}
	}

}

Mat HorizontalSegmentation(Mat img, int* histH)
{
	int maxseg = 0, start_pos = 0, end_pos = img.rows - 1;

	int i = 0;
	while(i < img.rows)
	{
		if (histH[i] > 2)
		{
			int auxm = 1, st_aux = i;
			i++;
			while (i < img.rows && histH[i] > 2)
			{
				auxm++;
				i++;
			}
			if (auxm > maxseg)
			{
				maxseg = auxm;
				start_pos = st_aux;
				end_pos = i - 1;
			}
		}
		else
			i++;
	}

	if (start_pos >= 0 && end_pos < img.rows)
	{
		Mat dst(img.rows - (start_pos + img.rows - end_pos - 1), img.cols, CV_8UC1);

		int x = 0;
		for (i = 0; i < img.rows; i++)
		{
			if (i >= start_pos && i <= end_pos)
			{
				for (int j = 0; j < img.cols; j++)
				{
					dst.at<uchar>(x, j) = img.at<uchar>(i, j);
				}
				x++;
			}
		}
		return dst;
	}
}

Mat extractLetter(Mat img, int start_pos, int end_pos)
{
	Mat dst(img.rows, img.cols - (start_pos + img.cols - end_pos - 1), CV_8UC1);
	int x = 0;
	
	for (int j = 0; j < img.cols; j++)
	{
		if (j >= start_pos && j <= end_pos)
		{
			for (int i = 0; i < img.rows; i++)
			{
				dst.at<uchar>(i, x) = img.at<uchar>(i, j);
			}
			x++;
		}
	}
	resize(dst, dst, Size(IMG_SIZE, IMG_SIZE));
	return dst;
}

bool validLetter(Mat img, int* histV, int start_pos, int end_pos, bool last_valid)
{
	// excludes characters such as ":" after NOM and PRENOM, and strings "NOM" and "PRENOM" will be excluded 
	// because their letters are too close to each other. 
	int zero_pixels_left = 0;
	int zero_pixels_right = 0;

	if (start_pos < 0 && end_pos >= img.cols)
		return false;

	int i = start_pos - 1;
	int j = end_pos + 1;

	while (!histV[i] && i >= 0)
	{
		zero_pixels_left++;
		i--;
	}

	while (!histV[j] && j < img.cols)
	{
		zero_pixels_right++;
		j++;
	}

	if (zero_pixels_left < 3)
		return false;

	if (zero_pixels_left < 10 && !last_valid)
		return false;

	if (zero_pixels_right < 3)
		return false;

	return true;
}

void testVerticalSegmentation(Mat img, int* histV)
{
	int i = 0;
	int start_pos = 0;
	int end_pos = 0;
	bool last_valid = true;  // for exclusion of ":" character that is after NOM, PRENOM
	
	while (i < img.cols)
	{
		start_pos = i;
		if (i == 0 && histV[i])
		{
			i++;
			last_valid = false;
			while (i < img.cols && histV[i])
			{
				i++;
			}
		}
		else
		{
			if (histV[i])
			{
				int maxseg = 1;
				i++;
				while (i < img.cols && histV[i])
				{
					maxseg++;
					i++;
				}
				if (maxseg < 24 && maxseg > 1)
				{
					end_pos = i - 1;
					if (validLetter(img, histV, start_pos, end_pos, last_valid))
					{
						Mat dst = extractLetter(img, start_pos, end_pos);
						int* zones = (int*)malloc(NUM_ZONES * sizeof(int));
						printf("\n\nThe image of this letter has %d pixels", numberofPixels(dst));
						zoning(dst, zones);
						printf(" and has the following values for the %d zones: \n", NUM_ZONES);
						for (int z = 0; z < NUM_ZONES; z++)
							printf("%d ", zones[z]);
						imshow("litera", dst);
						waitKey();
					}
					last_valid = validLetter(img, histV, start_pos, end_pos, last_valid);
				}
			}
			else
			{
				i++;
			}
		}
	}
}

void testSegmentation()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_GRAYSCALE);

		printf("\n\nimage size: %dx%d\n\n", img.cols, img.rows);

		Mat hist1(img.rows, img.cols, CV_8UC1);
		hist1.setTo(255);
		Mat dst = binarization(img, 180);

		imshow("bin", dst);
		imshow("input", img);

		int* histH = (int*)malloc(img.rows * sizeof(int));

		horizontal_hist(dst, histH);

		printf("\n\nBefore horizontal segmentation: \n");
		for (int i = 0; i < img.rows; i++)
		{
			printf("%d ", histH[i]);
			for (int j = 0; j < histH[i]; j++)
			{
				hist1.at<uchar>(i, j) = 0;
			}
		}

		imshow("hist h", hist1);

		Mat hdst = HorizontalSegmentation(dst, histH);
		horizontal_hist(hdst, histH);
	
		printf("\n\nAfter horizontal segmentation: \n");
		for (int i = 0; i < hdst.rows; i++)
			printf("%d ", histH[i]);

		imshow("horizontal", hdst);

		int* histV = (int*)malloc(hdst.cols * sizeof(int));
		vertical_hist(hdst, histV);
		Mat hist2(hdst.rows, hdst.cols, CV_8UC1);
		hist2.setTo(255);

		printf("\n\nVertical histogram after horizontal segmentation: \n");
		for (int j = 0; j < hdst.cols; j++)
		{
			printf("%d ", histV[j]);
			for (int i = hdst.rows - 1; i > hdst.rows - 1 - histV[j]; i--)
			{
				hist2.at<uchar>(i, j) = 0;
			}
		}
		imshow("hist v", hist2);

		testVerticalSegmentation(hdst, histV);

		waitKey();
	}
}

vector<Mat> VerticalSegmentation(Mat img, int* histV)
{
	vector<Mat> letters;
	int i = 0;
	int start_pos = 0;
	int end_pos = 0;
	bool last_valid = true;  // for exclusion of ":" character that is after NOM, PRENOM

	while (i < img.cols)
	{
		start_pos = i;
		if (i == 0 && histV[i])
		{
			i++;
			while (i < img.cols && histV[i])
			{
				i++;
			}
		}
		else
		{
			if (histV[i])
			{
				int maxseg = 1;
				i++;
				while (i < img.cols && histV[i])
				{
					maxseg++;
					i++;
				}
				if (maxseg < 24 && maxseg > 1)
				{
					end_pos = i - 1;
					if (validLetter(img, histV, start_pos, end_pos, last_valid))
					{
						Mat dst = extractLetter(img, start_pos, end_pos);
						letters.push_back(dst);
					}
					last_valid = validLetter(img, histV, start_pos, end_pos, last_valid);
				}
			}
			else
			{
				i++;
			}
		}
	}

	return letters;
}

vector<Mat> Segmentation(Mat img)
{
	vector<Mat> letters;

	int* histV = (int*)malloc(img.cols * sizeof(int));
	int* histH = (int*)malloc(img.rows * sizeof(int));

	horizontal_hist(img, histH);

	Mat hdst = HorizontalSegmentation(img, histH);
	horizontal_hist(hdst, histH);

	vertical_hist(hdst, histV);

	letters = VerticalSegmentation(hdst, histV);

	return letters;
}

int train(Mat X, Mat y)
{
	char fname[MAX_PATH];
	int index = 1, pos = 0, nrLetters = 0;
	vector<Mat> letters;

	FILE* f = fopen("dataset/written_name_train_v2.csv", "r");
	FILE* g = fopen("dataset/FeatureMatrix.csv", "w");
	char line[1024];

	if (!f) {
		printf("\nCan't open file (train).\n");
		return 0;
	}

	fgets(line, 1024, f);  // first line represent column name

	while (1)
	{
		sprintf(fname, "dataset/train_v2/train/TRAIN_%05d.jpg", index++);

		Mat img = imread(fname, IMREAD_GRAYSCALE);

		if (img.cols == 0) break;

		img = binarization(img, 180);

		fgets(line, 1024, f);

		char* name = strtok(line, ",");
		name = strtok(NULL, "\n");

		if (numberofPixels(img) > 10 && name != NULL)   // exclude blank photos and NULL names from csv
		{
			letters = Segmentation(img);

			if (letters.size() == strlen(name))
			{
				nrLetters += letters.size();
				for (int i = 0; i < letters.size(); i++)
				{
					int* histV = (int*)malloc(letters[i].cols * sizeof(int));
					int* histH = (int*)malloc(letters[i].rows * sizeof(int));

					vertical_hist(letters[i], histV);
					horizontal_hist(letters[i], histH);

					int* zones = (int*)malloc(NUM_ZONES * sizeof(int));
					zoning(letters[i], zones);

					for (int j = 0; j < X.cols; j++)
					{
						if(j < IMG_SIZE)
							X.at<int>(pos, j) = histV[j];
						else
							if(j < 2 * IMG_SIZE)
								X.at<int>(pos, j) = histH[j - IMG_SIZE];
							else
								X.at<int>(pos, j) = zones[j - 2 * IMG_SIZE];
						fprintf(g, "%d ", X.at<int>(pos, j));
					}

					y.at<int>(pos) = int(name[i]);
					fprintf(g, "%d\n", y.at<int>(pos));
					pos++;
				}
			}
		}
	}
	printf("\nTrain complete!\n");
	fclose(f);
	fclose(g);
	return nrLetters;
}

int readFeaturesFromCSV(Mat X, Mat y)
{
	int nrLetters = 0;
	FILE* f = fopen("dataset/FeatureMatrix.csv", "r");
	char line[1024];
	char* token;

	if (!f) {
		printf("\nCan't open file (FeatureMatrix.csv).\n");
		return 0;
	}

	while (fgets(line, 1024, f) != NULL)
	{
		token = strtok(line, " ");
		for (int j = 0; j < IMG_SIZE * 2 + NUM_ZONES - 1; j++)
		{
			X.at<int>(nrLetters, j) = atoi(token);
			token = strtok(NULL, " ");
		}
		X.at<int>(nrLetters, IMG_SIZE * 2 + NUM_ZONES - 1) = atoi(token);
		token = strtok(NULL, "\n");
		y.at<int>(nrLetters) = atoi(token);
		nrLetters++;
	}
	
	fclose(f);
	return nrLetters;
}

Mat takeBestKlines(Mat input_mat, int K)
{
	Mat output_mat = input_mat.clone();

	for (int i = 0; i < K; i++)
	{
		int min_idx = i;
		for (int j = i + 1; j < input_mat.rows; j++)
		{
			if (output_mat.at<float>(j, 0) < output_mat.at<float>(min_idx, 0))
			{
				min_idx = j;
			}
		}

		float temp_d = output_mat.at<float>(i, 0);
		float temp_c = output_mat.at<float>(i, 1);
		output_mat.at<float>(i, 0) = output_mat.at<float>(min_idx, 0);
		output_mat.at<float>(i, 1) = output_mat.at<float>(min_idx, 1);
		output_mat.at<float>(min_idx, 0) = temp_d;
		output_mat.at<float>(min_idx, 1) = temp_c;
	}

	return output_mat;
}

Mat compute_distances(Mat X, Mat y, int* histV, int* histH, int *zones, int nrLetters, int K)
{
	Mat distances(nrLetters, 2, CV_32FC1);
	distances.setTo(0);

	for (int i = 0; i < nrLetters; i++)
	{
		distances.at<float>(i, 1) = -1;
	}

	for (int i = 0; i < nrLetters; i++) {
		float distance = 0;
		for (int j = 0; j < X.cols; j++) 
		{
			if(j < IMG_SIZE)
				distance += (abs(histV[j] - X.at<int>(i, j)));
			else
				if (j < 2 * IMG_SIZE)
					distance += (abs(histH[j - IMG_SIZE] - X.at<int>(i, j)));
				else
					distance += (abs(zones[j - 2 * IMG_SIZE] - X.at<int>(i, j)));
		}
		distances.at<float>(i, 0) = distance;
		distances.at<float>(i, 1) = (float)y.at<int>(i);
	}

	distances = takeBestKlines(distances, K);

	return distances;
}

int find_closset_neighbor(int K, Mat input_mat)
{
	int maxVotes = 0;
	int closest_neighbor = 0;
	Mat votes(NRCLASSES, 1, CV_8UC1); 
	votes.setTo(0);

	for (int i = 0; i < K; i++)
	{
		votes.at<uchar>((int)input_mat.at<float>(i, 1))++;
	}

	for (int i = 0; i < votes.rows; i++) {
		if (votes.at<uchar>(i) > maxVotes) {
			maxVotes = votes.at<uchar>(i);
			closest_neighbor = i;
		}
	}

	return closest_neighbor;
}

void test_input_image()
{
	char fname[MAX_PATH];
	vector<Mat> letters;
	Mat X(1600000, IMG_SIZE * 2 + NUM_ZONES, CV_32SC1);
	Mat y(1600000, 1, CV_32SC1);
	int K = 0;
	int nrLetters = 0;

	printf("K = ");
	scanf("%d", &K);

	printf("\nWait! Reading from FeatureMatrix.csv\n");
	nrLetters = readFeaturesFromCSV(X, y);
	if (!nrLetters)
	{
		printf("\nWait! The training process takes a while.\n");
		nrLetters = train(X, y);
	}

	if (!nrLetters)
	{
		printf("FAIL!");
		return;
	}

	printf("\nFeature matrix has features for %d letters.", nrLetters);

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_GRAYSCALE);

		img = binarization(img, 180);

		Mat Kelem(K, 2, CV_32FC1);  // first col - distance, second col - class

		letters = Segmentation(img);
		printf("\n\nTest image - found letters: %d", letters.size());
		printf("\nWait! Recognizing each letter takes a while.\n");
		printf("\nI think is: ");

		for (int i = 0; i < letters.size(); i++)
		{
			int* histV = (int*)malloc(letters[i].cols * sizeof(int));
			int* histH = (int*)malloc(letters[i].rows * sizeof(int));

			vertical_hist(letters[i], histV);
			horizontal_hist(letters[i], histH);

			int* zones = (int*)malloc(NUM_ZONES * sizeof(int));
			zoning(letters[i], zones);

			Kelem = compute_distances(X, y, histV, histH, zones, nrLetters, K);

			int closest_neighbor = find_closset_neighbor(K, Kelem);

			printf("%c", (char)(closest_neighbor));
		}

		printf("\n\nDone!\n");

		imshow("input", img);
		waitKey();
	}
}

void test_images()
{
	char fname[MAX_PATH];
	vector<Mat> letters;
	Mat X(1600000, IMG_SIZE * 2 + NUM_ZONES, CV_32SC1);
	Mat y(1600000, 1, CV_32SC1);
	Mat C(NRCLASSES, NRCLASSES, CV_32FC1);  //confusion matrix
	C.setTo(0);
	int K = 0, index = 1, same = 0;
	int ok = 1;
	int nrTest = 0;
	int nrTestLetters = 0;
	int nrLetters = 0;
	float acc = 0;

	printf("K = ");
	scanf("%d", &K);
	printf("Insert the number of test images (up to 41370)  ");
	scanf("%d", &nrTest);

	FILE* f = fopen("dataset/written_name_test_v2.csv", "r");
	char line[1024];

	if (nrTest < 0 || nrTest > 41370) {
		printf("\nWrong input!\n");
		return;
	}

	if (!f) {
		printf("\nCan't open file (test).\n");
		return;
	}

	printf("\nWait! Reading from FeatureMatrix.csv\n");
	nrLetters = readFeaturesFromCSV(X, y);
	if (!nrLetters)
	{
		printf("\nWait! The training process takes a while.\n");
		nrLetters = train(X, y);
	}

	if (!nrLetters)
	{
		printf("FAIL!");
		return;
	}

	printf("\nFeature matrix has features for %d letters.\n", nrLetters);

	fgets(line, 1024, f);  // first line represent column name

	while (ok) {
		FILE* g = fopen("dataset/Results.csv", "w");
		printf("\nWait! The testing process takes a while.\n");

		while (index <= nrTest)
		{
			sprintf(fname, "dataset/test_v2/test/TEST_%04d.jpg", index++);

			Mat img = imread(fname, IMREAD_GRAYSCALE);

			if (img.cols == 0) break;

			img = binarization(img, 180);

			fgets(line, 1024, f);

			char* name = strtok(line, ",");

			printf("Processing image %s...\n", name);

			name = strtok(NULL, "\n");

			if (numberofPixels(img) > 10 && name != NULL)  // exclude blank photos and NULL names from csv
			{
				letters = Segmentation(img);

				if (letters.size() == strlen(name))
				{
					Mat Kelem(K, 2, CV_32FC1);  // first col - distance, second col - class
					nrTestLetters += letters.size();
					bool ok = true;

					for (int i = 0; i < letters.size(); i++)
					{
						int* histV = (int*)malloc(letters[i].cols * sizeof(int));
						int* histH = (int*)malloc(letters[i].rows * sizeof(int));

						vertical_hist(letters[i], histV);
						horizontal_hist(letters[i], histH);

						int* zones = (int*)malloc(NUM_ZONES * sizeof(int));
						zoning(letters[i], zones);

						Kelem = compute_distances(X, y, histV, histH, zones, nrLetters, K);

						int closest_neighbor = find_closset_neighbor(K, Kelem);

						C.at<int>((int)(name[i]), closest_neighbor)++;
						if ((char)(closest_neighbor) != name[i])
							same++;
					}
				}
			}

		}
		printf("\nTest complete!\n");
		fprintf(g, "Confusion matrix:\n");
		for (int i = 0; i < NRCLASSES; i++) {
			for (int j = 0; j < NRCLASSES; j++) {
				fprintf(g, "%d ", C.at<int>(i, j));
			}
			fprintf(g, "\n");
		}
		printf("\nThe confusion matrix was written in the file. File path: dataset/Results.csv\n");
		printf("\n%d letters were identified in the test set.\n", nrTestLetters);
		printf("%d letters were identified correctly.\n", same);
		fprintf(g, "\nFeature matrix has features for %d letters.\n", nrLetters);
		fprintf(g, "%d letters were identified from %d test images.\n", nrTestLetters, nrTest);
		fprintf(g, "%d letters were identified correctly.\n", same);
		if (!nrTestLetters)
		{
			printf("\nAccuracy: N/A \n");
			fprintf(g, "\nFor K = %d, accuracy = N/A.\n", K);
		}
		else
		{
			acc = (float)same / nrTestLetters;
			printf("\nAccuracy: %f\n", acc);
			fprintf(g, "\nFor K = %d, accuracy =  %f.\n", K, acc);
		}
		printf("\nDone!\n");
		fclose(g);
		printf("\n0 to exit\n");
		scanf("%d", &ok);
	}
	fclose(f);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("1 - Test letter segmentation and zoning\n");
		printf("2 - Test on input image\n");
		printf("3 - Test on set of test images\n");
		printf("0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testSegmentation();
				break;
			case 2:
				test_input_image();
				break;
			case 3:
				test_images();
				break;
		}
	}
	while (op!=0);
	return 0;
}