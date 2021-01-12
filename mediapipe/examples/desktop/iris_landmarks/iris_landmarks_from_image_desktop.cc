// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A utility to extract iris depth from a single image of face using the graph
// mediapipe/graphs/iris_tracking/iris_depth_cpu.pbtxt.
#include <cstdlib>
#include <memory>
#include <fstream>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_image_bytes";
// constexpr char kOutputImageStream[] = "output_image";
constexpr char kLeftIrisLandmarksStream[] = "left_iris_landmarks";
constexpr char kRightIrisLandmarksStream[] = "right_iris_landmarks";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kCalculatorGraphConfigFile[] =
    "mediapipe/graphs/iris_tracking/iris_landmarks_cpu.pbtxt";
constexpr float kMicrosPerSecond = 1e6;

DEFINE_string(input_image_path, "",
              "Full path of image to load. "
              "If not provided, nothing will run.");

namespace {

::mediapipe::StatusOr<std::string> ReadFileToString(
    const std::string& file_path) {
  std::string contents;
  MP_RETURN_IF_ERROR(::mediapipe::file::GetContents(file_path, &contents));
  return contents;
}

::mediapipe::Status ProcessImage(
    std::unique_ptr<::mediapipe::CalculatorGraph> graph) {
  LOG(INFO) << "Load the image.";
  ASSIGN_OR_RETURN(const std::string raw_image,
                   ReadFileToString(FLAGS_input_image_path));

  LOG(INFO) << "Start running the calculator graph.";
  // ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller output_image_poller,
  //                  graph->AddOutputStreamPoller(kOutputImageStream));
  ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller left_iris_landmarks_poller,
                   graph->AddOutputStreamPoller(kLeftIrisLandmarksStream));
  ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller right_iris_landmarks_poller,
                   graph->AddOutputStreamPoller(kRightIrisLandmarksStream));
  MP_RETURN_IF_ERROR(graph->StartRun({}));

  // Send image packet into the graph.
  const size_t fake_timestamp_us = (double)cv::getTickCount() /
                                   (double)cv::getTickFrequency() *
                                   kMicrosPerSecond;
  MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
      kInputStream, ::mediapipe::MakePacket<std::string>(raw_image).At(
                        ::mediapipe::Timestamp(fake_timestamp_us))));

  // Get the graph result packets, or stop if that fails.
  ::mediapipe::Packet left_iris_landmarks_packet;
  if (!left_iris_landmarks_poller.Next(&left_iris_landmarks_packet)) {
    return ::mediapipe::UnknownError(
        "Failed to get packet from output stream 'left_iris_landmarks'.");
  }
  const auto& left_iris_landmarks = left_iris_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();
  // const int left_iris_depth_cm = std::round(left_iris_depth_mm / 10);
  // std::cout << "Left Iris Depth: " << left_iris_depth_cm << " cm." << std::endl;

  ::mediapipe::Packet right_iris_landmarks_packet;
  if (!right_iris_landmarks_poller.Next(&right_iris_landmarks_packet)) {
    return ::mediapipe::UnknownError(
        "Failed to get packet from output stream 'right_iris_landmarks'.");
  }
  const auto& right_iris_landmarks = right_iris_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();
  // const int right_iris_depth_cm = std::round(right_iris_depth_mm / 10);
  // std::cout << "Right Iris Depth: " << right_iris_depth_cm << " cm."
  //           << std::endl;

  // LOG(INFO) << "Left Iris:\n";
  // LOG(INFO) << "Center    " << left_iris_landmarks.landmark(0).x() << '\t' 
  //                           << left_iris_landmarks.landmark(0).y() << '\n';
  // LOG(INFO) << "Top       " << left_iris_landmarks.landmark(2).x() << '\t' 
  //                           << left_iris_landmarks.landmark(2).y() << '\n';
  // LOG(INFO) << "Bottom    " << left_iris_landmarks.landmark(4).x() << '\t' 
  //                           << left_iris_landmarks.landmark(4).y() << '\n';
  // LOG(INFO) << "Left      " << left_iris_landmarks.landmark(3).x() << '\t' 
  //                           << left_iris_landmarks.landmark(3).y() << '\n';
  // LOG(INFO) << "Right     " << left_iris_landmarks.landmark(1).x() << '\t' 
  //                           << left_iris_landmarks.landmark(1).y() << '\n';

  // LOG(INFO) << "Right Iris:\n";
  // LOG(INFO) << "Center    " << right_iris_landmarks.landmark(0).x() << '\t' 
  //                           << right_iris_landmarks.landmark(0).y() << '\n';
  // LOG(INFO) << "Top       " << right_iris_landmarks.landmark(2).x() << '\t' 
  //                           << right_iris_landmarks.landmark(2).y() << '\n';
  // LOG(INFO) << "Bottom    " << right_iris_landmarks.landmark(4).x() << '\t' 
  //                           << right_iris_landmarks.landmark(4).y() << '\n';
  // LOG(INFO) << "Left      " << right_iris_landmarks.landmark(3).x() << '\t' 
  //                           << right_iris_landmarks.landmark(3).y() << '\n';
  // LOG(INFO) << "Right     " << right_iris_landmarks.landmark(1).x() << '\t' 
  //                           << right_iris_landmarks.landmark(1).y() << '\n';

// append iris landmarks to file
std::ofstream outFile;
LOG(INFO) << "Writing Landmarks to File...\n";
outFile.open("../jupyter-notebook/dataset/labels/iris_landmarks.csv", std::ios::app);

if (!outFile.is_open()) {

  LOG(INFO) << "File Cannot Be Opened!";

} else {

  outFile << std::to_string(left_iris_landmarks.landmark(0).x()) + ','
            + std::to_string(left_iris_landmarks.landmark(0).y()) + ','
            + std::to_string(left_iris_landmarks.landmark(2).x()) + ','
            + std::to_string(left_iris_landmarks.landmark(2).y()) + ','
            + std::to_string(left_iris_landmarks.landmark(4).x()) + ','
            + std::to_string(left_iris_landmarks.landmark(4).y()) + ','
            + std::to_string(left_iris_landmarks.landmark(3).x()) + ','
            + std::to_string(left_iris_landmarks.landmark(3).y()) + ','
            + std::to_string(left_iris_landmarks.landmark(1).x()) + ','
            + std::to_string(left_iris_landmarks.landmark(1).y()) + ','
            + std::to_string(right_iris_landmarks.landmark(0).x()) + ','
            + std::to_string(right_iris_landmarks.landmark(0).y()) + ','
            + std::to_string(right_iris_landmarks.landmark(2).x()) + ','
            + std::to_string(right_iris_landmarks.landmark(2).y()) + ','
            + std::to_string(right_iris_landmarks.landmark(4).x()) + ','
            + std::to_string(right_iris_landmarks.landmark(4).y()) + ','
            + std::to_string(right_iris_landmarks.landmark(3).x()) + ','
            + std::to_string(right_iris_landmarks.landmark(3).y()) + ','
            + std::to_string(right_iris_landmarks.landmark(1).x()) + ','
            + std::to_string(right_iris_landmarks.landmark(1).y()) + '\n';

  outFile.close();

}



std::cout << "Output:\n" + std::to_string(left_iris_landmarks.landmark(0).x()) + ','
          + std::to_string(left_iris_landmarks.landmark(0).y()) + ','
          + std::to_string(left_iris_landmarks.landmark(2).x()) + ','
          + std::to_string(left_iris_landmarks.landmark(2).y()) + ','
          + std::to_string(left_iris_landmarks.landmark(4).x()) + ','
          + std::to_string(left_iris_landmarks.landmark(4).y()) + ','
          + std::to_string(left_iris_landmarks.landmark(3).x()) + ','
          + std::to_string(left_iris_landmarks.landmark(3).y()) + ','
          + std::to_string(left_iris_landmarks.landmark(1).x()) + ','
          + std::to_string(left_iris_landmarks.landmark(1).y()) + ','
          + std::to_string(right_iris_landmarks.landmark(0).x()) + ','
          + std::to_string(right_iris_landmarks.landmark(0).y()) + ','
          + std::to_string(right_iris_landmarks.landmark(2).x()) + ','
          + std::to_string(right_iris_landmarks.landmark(2).y()) + ','
          + std::to_string(right_iris_landmarks.landmark(4).x()) + ','
          + std::to_string(right_iris_landmarks.landmark(4).y()) + ','
          + std::to_string(right_iris_landmarks.landmark(3).x()) + ','
          + std::to_string(right_iris_landmarks.landmark(3).y()) + ','
          + std::to_string(right_iris_landmarks.landmark(1).x()) + ','
          + std::to_string(right_iris_landmarks.landmark(1).y()) + '\n';







  // ::mediapipe::Packet output_image_packet;
  // if (!output_image_poller.Next(&output_image_packet)) {
  //   return ::mediapipe::UnknownError(
  //       "Failed to get packet from output stream 'output_image'.");
  // }
  // auto& output_frame = output_image_packet.Get<::mediapipe::ImageFrame>();

  // Convert back to opencv for display or saving.
  // cv::Mat output_frame_mat = ::mediapipe::formats::MatView(&output_frame);
  // std::cout << "width: " << output_frame_mat.size().width << "\theight: " << output_frame_mat.size().height << '\n';
  // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
  // const bool save_image = !FLAGS_output_image_path.empty();
  // if (save_image) {
  //   LOG(INFO) << "Saving image to file...";
  //   cv::imwrite(FLAGS_output_image_path, output_frame_mat);
  // } else {
  //   cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  //   cv::imshow(kWindowName, output_frame_mat);
  //   // Press any key to exit.
  //   cv::waitKey(0);
  // }

  LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph->CloseInputStream(kInputStream));
  return graph->WaitUntilDone();
}

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(::mediapipe::file::GetContents(
      kCalculatorGraphConfigFile, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  ::mediapipe::CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  std::unique_ptr<::mediapipe::CalculatorGraph> graph =
      absl::make_unique<::mediapipe::CalculatorGraph>();
  MP_RETURN_IF_ERROR(graph->Initialize(config));

  const bool load_image = !FLAGS_input_image_path.empty();
  if (load_image) {
    return ProcessImage(std::move(graph));
  } else {
    return ::mediapipe::InvalidArgumentError("Missing image file.");
  }
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
