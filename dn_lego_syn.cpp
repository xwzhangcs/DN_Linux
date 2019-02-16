#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <experimental/filesystem>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"

namespace fs = std::experimental::filesystem;

int const max_BINARY_value = 255;

cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight);

int find_threshold(cv::Mat src, bool bground);

double readNumber(const rapidjson::Value& node, const char* key, double default_value) {
	if (node.HasMember(key) && node[key].IsDouble()) {
		return node[key].GetDouble();
	}
	else if (node.HasMember(key) && node[key].IsInt()) {
		return node[key].GetInt();
	}
	else {
		return default_value;
	}
}

std::vector<double> read1DArray(const rapidjson::Value& node, const char* key) {
	std::vector<double> array_values;
	if (node.HasMember(key)) {
		const rapidjson::Value& data = node[key];
		array_values.resize(data.Size());
		for (int i = 0; i < data.Size(); i++)
			array_values[i] = data[i].GetDouble();
		return array_values;
	}
	else {
		return array_values;
	}
}

bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value) {
	if (node.HasMember(key) && node[key].IsBool()) {
		return node[key].GetBool();
	}
	else {
		return default_value;
	}
}

std::string readStringValue(const rapidjson::Value& node, const char* key) {
	if (node.HasMember(key) && node[key].IsString()) {
		return node[key].GetString();
	}
	else {
		throw "Could not read string from node";
	}
}

void process_single_chip(std::string metajson, std::string modeljson);

int main(int argc, const char* argv[]) {
	if (argc != 3) {
	std::cerr << "usage: app <path-to-image-JSON-file> <path-to-model-config-JSON-file>\n";
	return -1;
	}
	for (const auto & entry : fs::directory_iterator(argv[1])) {
		std::cout << entry.path() << std::endl;
	}
	return 0;
}

void process_single_chip(std::string metajson, std::string modeljson) {
	// read image json file
	FILE* fp = fopen(metajson.c_str(), "r"); // non-Windows use "r"
	char readBuffer[10240];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document doc;
	doc.ParseStream(is);
	// size of chip
	std::vector<double> facChip_size = read1DArray(doc, "size");
	// ground
	bool bground = readBoolValue(doc, "ground", false);
	// image file
	std::string img_name = readStringValue(doc, "imagename");
	// score
	double score = readNumber(doc, "score", 0.2);
	fclose(fp);
	// first decide whether it's a valid chip
	bool bvalide = true;
	if (facChip_size[0] < 5.0 && facChip_size[1] < 5.0)
		bvalide = false;
	else if (facChip_size[0] < 30.0 && facChip_size[1] < 30.0 && score < 0.8)
		bvalide = false;
	else if (facChip_size[0] > 30.0 && facChip_size[1] < 30.0 && score < 0.7)
		bvalide = false;
	else if (facChip_size[0] < 30.0 && facChip_size[1] > 30.0 && score < 0.7)
		bvalide = false;
	else if (facChip_size[0] > 30.0 && facChip_size[1] > 30.0 && score < 0.3)
		bvalide = false;
	else {
		bvalide = true;
	}
	if (!bvalide) {
		// write back to json file
		fp = fopen(metajson.c_str(), "w"); // non-Windows use "w"
		rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
		doc.AddMember("valid", bvalide, alloc);
		// compute avg color
		cv::Scalar avg_color(0, 0, 0);
		cv::Mat src = cv::imread(img_name, 1);
		for (int i = 0; i < src.size().height; i++) {
			for (int j = 0; j < src.size().width; j++) {
				avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
				avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
				avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
			}
		}
		avg_color.val[0] = avg_color.val[0] / (src.size().height * src.size().width);
		avg_color.val[1] = avg_color.val[1] / (src.size().height * src.size().width);
		avg_color.val[2] = avg_color.val[2] / (src.size().height * src.size().width);

		rapidjson::Value avg_color_json(rapidjson::kArrayType);
		avg_color_json.PushBack(avg_color.val[0], alloc);
		avg_color_json.PushBack(avg_color.val[1], alloc);
		avg_color_json.PushBack(avg_color.val[2], alloc);
		doc.AddMember("bg_color", avg_color_json, alloc);

		char writeBuffer[10240];
		rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
		rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
		doc.Accept(writer);
		fclose(fp);
		return;
	}
	int type = 0;
	if (facChip_size[0] < 30.0 && facChip_size[1] < 30.0 && score > 0.8)
		type = 1;
	else if (facChip_size[0] > 30.0 && facChip_size[1] < 30.0 && score > 0.7)
		type = 2;
	else if (facChip_size[0] < 30.0 && facChip_size[1] > 30.0 && score > 0.7)
		type = 3;
	else if (facChip_size[0] > 30.0 && facChip_size[1] > 30.0 && score > 0.3)
		type = 4;
	else {
		// do nothing
	}
	// read model config json file
	fp = fopen(modeljson.c_str(), "r"); // non-Windows use "r"
	memset(readBuffer, 0, sizeof(readBuffer));
	rapidjson::FileReadStream isModel(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document docModel;
	docModel.ParseStream(isModel);
	std::string model_name;
	std::string grammar_name;
	if (bground) { // choose grammar2
		grammar_name = "grammar2";
	}
	else { // choose grammar1
		grammar_name = "grammar1";
	}
	rapidjson::Value& grammar = docModel[grammar_name.c_str()];
	// path of DN model
	model_name = readStringValue(grammar, "model");
	std::cout << "model_name is " << model_name << std::endl;
	// number of paras
	int num_paras = readNumber(grammar, "number_paras", 5);
	std::cout << "num_paras is " << num_paras << std::endl;
	// range of Rows
	std::vector<double> tmp_array = read1DArray(grammar, "rangeOfRows");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the rangeOfRows member in the JSON file" << std::endl;
		return;
	}
	std::pair<int, int> imageRows(tmp_array[0], tmp_array[1]);
	std::cout << "imageRows is " << imageRows.first << ", " << imageRows.second << std::endl;
	// range of Cols
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfCols");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the rangeOfCols member in the JSON file" << std::endl;
		return;
	}
	std::pair<int, int> imageCols(tmp_array[0], tmp_array[1]);
	std::cout << "imageCols is " << imageCols.first << ", " << imageCols.second << std::endl;
	// range of Grouping
	tmp_array.empty();
	tmp_array = read1DArray(grammar, "rangeOfGrouping");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the rangeOfGrouping member in the JSON file" << std::endl;
		return;
	}
	std::pair<int, int> imageGroups(tmp_array[0], tmp_array[1]);
	std::cout << "imageGroups is " << imageGroups.first << ", " << imageGroups.second << std::endl;
	// default size for NN
	int height = 224; // DNN image height
	int width = 224; // DNN image width
	tmp_array.empty();
	tmp_array = read1DArray(docModel, "defaultSize");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the defaultSize member in the JSON file" << std::endl;
		return;
	}
	width = tmp_array[0];
	height = tmp_array[1];
	std::cout << "width is " << width << std::endl;
	std::cout << "height is " << height << std::endl;
	std::pair<int, int> imageDoors(2, 6);
	if (bground) {
		tmp_array.empty();
		tmp_array = read1DArray(grammar, "rangeOfDoors");
		if (tmp_array.size() != 2) {
			std::cout << "Please check the rangeOfDoors member in the JSON file" << std::endl;
			return;
		}
		imageDoors.first = tmp_array[0];
		imageDoors.second = tmp_array[1];
		std::cout << "imageDoors is " << imageDoors.first << ", " << imageDoors.second << std::endl;
	}
	tmp_array.empty();
	tmp_array = read1DArray(docModel, "targetChipSize");
	if (tmp_array.size() != 2) {
		std::cout << "Please check the targetChipSize member in the JSON file" << std::endl;
		return;
	}
	double target_width = tmp_array[0];
	double target_height = tmp_array[1];
	// get chips folder path
	std::string chips_folder = readStringValue(docModel, "chipFolder");
	// get chips folder path
	std::string segs_folder = readStringValue(docModel, "segsFolder");
	// get dnn folder path
	std::string dnn_folder = readStringValue(docModel, "dnnFolder");
	fclose(fp);
	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_name);
	module->to(at::kCUDA);

	assert(module != nullptr);
	std::cout << "ok\n";

	// reshape the chip and pick the representative one
	double ratio_width, ratio_height;
	// image relative name
	std::size_t found = img_name.find_first_of("image/");
	if (found < 0) {
		std::cout << "found failed!!!" << std::endl;
		return;
	}
	found = found + 6;
	cv::Mat src_chip, dst_chip, croppedImage;
	if (type == 1) {
		src_chip = cv::imread(img_name);
		ratio_width = target_width / facChip_size[0] - 1;
		ratio_height = target_height / facChip_size[1] - 1;
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		croppedImage = dst_chip;
		cv::imwrite(chips_folder + "/" + img_name.substr(found + 1), croppedImage);
	}
	else if (type == 2) {
		src_chip = cv::imread(img_name);
		int times = ceil(facChip_size[0] / target_width);
		ratio_width = (times * target_width - facChip_size[0]) / facChip_size[0];
		ratio_height = target_height / facChip_size[1] - 1;
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		// crop 30 * 30
		croppedImage = dst_chip(cv::Rect(dst_chip.size().width * 0.1, 0, dst_chip.size().width / times, dst_chip.size().height));
		cv::imwrite(chips_folder + "/" + img_name.substr(found + 1), croppedImage);
	}
	else if (type == 3) {
		src_chip = cv::imread(img_name);
		int times = ceil(facChip_size[1] / target_height);
		ratio_height = (times * target_height - facChip_size[1]) / facChip_size[1];
		ratio_width = target_width / facChip_size[0] - 1;
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		// crop 30 * 30
		croppedImage = dst_chip(cv::Rect(0, dst_chip.size().height * (times - 1) / times, dst_chip.size().width, dst_chip.size().height / times));
		cv::imwrite(chips_folder + "/" + img_name.substr(found + 1), croppedImage);
	}
	else if (type == 4) {
		src_chip = cv::imread(img_name);
		int times_width = ceil(facChip_size[0] / target_width);
		int times_height = ceil(facChip_size[1] / target_height);
		ratio_width = (times_width * target_width - facChip_size[0]) / facChip_size[0];
		ratio_height = (times_height * target_height - facChip_size[1]) / facChip_size[1];
		std::cout << "ratio_width is " << ratio_width << std::endl;
		std::cout << "ratio_height is " << ratio_height << std::endl;
		int top = (int)(ratio_height * src_chip.rows);
		int bottom = 0;
		int left = 0;
		int right = (int)(ratio_width * src_chip.cols);
		int borderType = cv::BORDER_REFLECT_101;
		cv::Scalar value(0, 0, 0);
		cv::copyMakeBorder(src_chip, dst_chip, top, bottom, left, right, borderType, value);
		// crop 30 * 30
		croppedImage = dst_chip(cv::Rect(dst_chip.size().width * 0.1, dst_chip.size().height * (times_height - 1) / times_height, dst_chip.size().width / times_width, dst_chip.size().height / times_height));
		cv::imwrite(chips_folder + "/" + img_name.substr(found + 1), croppedImage);
	}
	else {
		// do nothing
	}
	// load image
	cv::Mat src, dst_ehist, dst_classify;
	//src = cv::imread(img_name, 1);
	src = croppedImage;
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	dst_ehist = bgr[2];
	// threshold classification
	int threshold = find_threshold(src, bground);
	std::wcout << "threshold is " << threshold << std::endl;
	cv::threshold(dst_ehist, dst_classify, threshold, max_BINARY_value, cv::THRESH_BINARY);
	// generate input image for DNN
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	cv::Mat scale_img;
	cv::resize(dst_classify, scale_img, cv::Size(width, height));
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(scale_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
	}
	cv::Mat dnn_img(scale_img.size(), CV_8UC3, bg_color);
	for (int i = 0; i< contours.size(); i++)
	{
		if (hierarchy[i][2] != -1) continue;
		cv::rectangle(dnn_img, cv::Point(boundRect[i].tl().x + 1, boundRect[i].tl().y + 1), cv::Point(boundRect[i].br().x, boundRect[i].br().y), window_color, -1);
	}
	cv::cvtColor(dnn_img, dnn_img, CV_BGR2RGB);
	cv::Mat img_float;
	dnn_img.convertTo(img_float, CV_32F, 1.0 / 255);
	auto img_tensor = torch::from_blob(img_float.data, { 1, 224, 224, 3 }).to(torch::kCUDA);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	torch::Tensor out_tensor = module->forward(inputs).toTensor();
	std::cout << out_tensor.slice(1, 0, num_paras) << std::endl;
	std::vector<double> paras;
	for (int i = 0; i < num_paras; i++) {
		paras.push_back(out_tensor.slice(1, i, i + 1).item<float>());
	}
	// predict img by DNN
	int img_rows = round(paras[0] * (imageRows.second - imageRows.first) + imageRows.first);
	int img_cols = round(paras[1] * (imageCols.second - imageCols.first) + imageCols.first);
	int img_groups = 1;
	double relative_width = paras[2];
	double relative_height = paras[3];


	// find the average color for window/non-window
	cv::Scalar bg_avg_color(0, 0, 0);
	cv::Scalar win_avg_color(0, 0, 0);
	{
		int bg_count = 0;
		int win_count = 0;
		for (int i = 0; i < dst_classify.size().height; i++) {
			for (int j = 0; j < dst_classify.size().width; j++) {
				if ((int)dst_classify.at<uchar>(i, j) == 0) {
					win_avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
					win_avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
					win_avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
					win_count++;
				}
				else {
					bg_avg_color.val[0] += src.at<cv::Vec3b>(i, j)[0];
					bg_avg_color.val[1] += src.at<cv::Vec3b>(i, j)[1];
					bg_avg_color.val[2] += src.at<cv::Vec3b>(i, j)[2];
					bg_count++;
				}
			}
		}
		win_avg_color.val[0] = win_avg_color.val[0] / win_count;
		win_avg_color.val[1] = win_avg_color.val[1] / win_count;
		win_avg_color.val[2] = win_avg_color.val[2] / win_count;

		bg_avg_color.val[0] = bg_avg_color.val[0] / bg_count;
		bg_avg_color.val[1] = bg_avg_color.val[1] / bg_count;
		bg_avg_color.val[2] = bg_avg_color.val[2] / bg_count;
	}
	// write back to json file
	fp = fopen(metajson.c_str(), "w"); // non-Windows use "w"
	rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();
	doc.AddMember("valid", bvalide, alloc);

	rapidjson::Value paras_json(rapidjson::kObjectType);
	paras_json.AddMember("rows", img_rows, alloc);
	paras_json.AddMember("cols", img_cols, alloc);
	paras_json.AddMember("grouping", img_groups, alloc);
	paras_json.AddMember("relativeWidth", relative_width, alloc);
	paras_json.AddMember("relativeHeight", relative_height, alloc);
	doc.AddMember("paras", paras_json, alloc);

	rapidjson::Value bg_color_json(rapidjson::kArrayType);
	bg_color_json.PushBack(bg_avg_color.val[0], alloc);
	bg_color_json.PushBack(bg_avg_color.val[1], alloc);
	bg_color_json.PushBack(bg_avg_color.val[2], alloc);
	doc.AddMember("bg_color", bg_color_json, alloc);

	rapidjson::Value win_color_json(rapidjson::kArrayType);
	win_color_json.PushBack(win_avg_color.val[0], alloc);
	win_color_json.PushBack(win_avg_color.val[1], alloc);
	win_color_json.PushBack(win_avg_color.val[2], alloc);
	doc.AddMember("window_color", win_color_json, alloc);

	char writeBuffer[10240];
	rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	doc.Accept(writer);
	fclose(fp);

	// for 
	cv::Mat syn_img = generateFacadeSynImage(width, height, img_rows, img_cols, img_groups, relative_width, relative_height);
	// recover to the original image
	cv::resize(syn_img, syn_img, src.size());
	cv::resize(dnn_img, dnn_img, src.size());
	for (int i = 0; i < syn_img.size().height; i++) {
		for (int j = 0; j < syn_img.size().width; j++) {
			if (syn_img.at<cv::Vec3b>(i, j)[0] == 0) {
				syn_img.at<cv::Vec3b>(i, j)[0] = win_avg_color.val[0];
				syn_img.at<cv::Vec3b>(i, j)[1] = win_avg_color.val[1];
				syn_img.at<cv::Vec3b>(i, j)[2] = win_avg_color.val[2];
			}
			else {
				syn_img.at<cv::Vec3b>(i, j)[0] = bg_avg_color.val[0];
				syn_img.at<cv::Vec3b>(i, j)[1] = bg_avg_color.val[1];
				syn_img.at<cv::Vec3b>(i, j)[2] = bg_avg_color.val[2];
			}
		}
	}
	cv::imwrite(dnn_folder + "/" + img_name.substr(found + 1), syn_img);

	for (int i = 0; i < dnn_img.size().height; i++) {
		for (int j = 0; j < dnn_img.size().width; j++) {
			if (dnn_img.at<cv::Vec3b>(i, j)[0] == 0) {
				dnn_img.at<cv::Vec3b>(i, j)[0] = win_avg_color.val[0];
				dnn_img.at<cv::Vec3b>(i, j)[1] = win_avg_color.val[1];
				dnn_img.at<cv::Vec3b>(i, j)[2] = win_avg_color.val[2];
			}
			else {
				dnn_img.at<cv::Vec3b>(i, j)[0] = bg_avg_color.val[0];
				dnn_img.at<cv::Vec3b>(i, j)[1] = bg_avg_color.val[1];
				dnn_img.at<cv::Vec3b>(i, j)[2] = bg_avg_color.val[2];
			}
		}
	}
	cv::imwrite(segs_folder + "/" + img_name.substr(found + 1), dnn_img);
}

int find_threshold(cv::Mat src, bool bground) {
	//Convert pixel values to other color spaces.
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> bgr;   //destination array
	cv::split(hsv, bgr);//split source 
	for (int i = 0; i < 3; i++)
		cv::equalizeHist(bgr[i], bgr[i]);
	/// Load an image
	cv::Mat src_gray = bgr[2];
	for (int threshold = 40; threshold < 160; threshold += 5) {
		cv::Mat dst;
		cv::threshold(src_gray, dst, threshold, max_BINARY_value, cv::THRESH_BINARY);
		int count = 0;
		for (int i = 0; i < dst.size().height; i++) {
			for (int j = 0; j < dst.size().width; j++) {
				//noise
				if ((int)dst.at<uchar>(i, j) == 0) {
					count++;
				}
			}
		}
		float percentage = count * 1.0 / (dst.size().height * dst.size().width);
		if (percentage > 0.25 && !bground)
			return threshold;
		if (percentage > 0.35 && bground)
			return threshold;
	}
}

cv::Mat generateFacadeSynImage(int width, int height, int imageRows, int imageCols, int imageGroups, double imageRelativeWidth, double imageRelativeHeight) {
	cv::Scalar bg_color(255, 255, 255); // white back ground
	cv::Scalar window_color(0, 0, 0); // black for windows
	int NR = imageRows;
	int NC = imageCols;
	int NG = imageGroups;
	double ratioWidth = imageRelativeWidth;
	double ratioHeight = imageRelativeHeight;
	int thickness = -1;
	cv::Mat result(height, width, CV_8UC3, bg_color);
	double FH = height * 1.0 / NR;
	double FW = width * 1.0 / NC;
	double WH = FH * ratioHeight;
	double WW = FW * ratioWidth;
	std::cout << "NR is " << NR << std::endl;
	std::cout << "NC is " << NC << std::endl;
	std::cout << "FH is " << FH << std::endl;
	std::cout << "FW is " << FW << std::endl;
	std::cout << "ratioWidth is " << ratioWidth << std::endl;
	std::cout << "ratioHeight is " << ratioHeight << std::endl;
	std::cout << "WH is " << WH << std::endl;
	std::cout << "WW is " << WW << std::endl;
	// draw facade image
	if (NG == 1) {
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = (FW - WW) * 0.5 + FW * j;
				float y1 = (FH - WH) * 0.5 + FH * i;
				float x2 = x1 + WW;
				float y2 = y1 + WH;
				cv::rectangle(result, cv::Point(std::round(x1), std::round(y1)), cv::Point(std::round(x2), std::round(y2)), window_color, thickness);
			}
		}
	}
	else {
		double GFW = WW / NG;
		double GWW = WW / NG - 2;
		for (int i = 0; i < NR; ++i) {
			for (int j = 0; j < NC; ++j) {
				float x1 = (FW - WW) * 0.5 + FW * j;
				float y1 = (FH - WH) * 0.5 + FH * i;
				for (int k = 0; k < NG; k++) {
					float g_x1 = x1 + GFW * k;
					float g_y1 = y1;
					float g_x2 = g_x1 + GWW;
					float g_y2 = g_y1 + WH;

					cv::rectangle(result, cv::Point(std::round(g_x1), std::round(g_y1)), cv::Point(std::round(g_x2), std::round(g_y2)), window_color, thickness);
				}
			}
		}
	}
	return result;
}