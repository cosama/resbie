#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstdint>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <unordered_map>
#include <utility>
#include <vector>

#define HASH_P 116101
#define MAX_N 10000000000

struct ConfigSetting {
  /* for point cloud pre-process */
  int useful_corner_num_ = 30;

  /* for key points */
  float plane_merge_normal_thre_ = 0.1f;
  float plane_merge_dis_thre_ = 0.3f;
  float plane_detection_thre_ = 0.01f;
  float voxel_size_ = 1.0f;
  int voxel_init_num_ = 10;
  int proj_plane_num_ = 3;
  float proj_image_resolution_ = 0.5f;
  float proj_image_high_inc_ = 0.5f;
  float proj_dis_min_ = 0.0f;
  float proj_dis_max_ = 5.0f;
  float summary_min_thre_ = 10.0f;

  /* for STD */
  float descriptor_near_num_ = 10.0f;
  float descriptor_min_len_ = 1.0f;
  float descriptor_max_len_ = 10.0f;
  float non_max_suppression_radius_ = 3.0f;
  float std_side_resolution_ = 0.2f;

  /* for place recognition */
  int skip_near_num_ = 20;
  int candidate_num_ = 50;
  float rough_dis_threshold_ = 0.03f;
  float similarity_threshold_ = 0.7f;
  float icp_threshold_ = 0.5f;
  float normal_threshold_ = 0.1f;
  float dis_threshold_ = 0.3f;
};

struct BinaryDescriptor {
  uint64_t occupy_bits_ = 0;
  unsigned char summary_ = 0;
  Eigen::Vector3d location_ = Eigen::Vector3d::Zero();
};

struct STD {
  Eigen::Vector3d triangle_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
  int frame_number_ = 0;
  BinaryDescriptor binary_A_;
  BinaryDescriptor binary_B_;
  BinaryDescriptor binary_C_;
};

struct BTCPlane {
  Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance_ = Eigen::Matrix3d::Zero();
  float d_ = 0.0f;
  int id_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
};

struct STDMatchList {
  std::vector<std::pair<STD, STD>> match_list_;
  std::pair<int, int> match_id_{0, 0};
};

class BTCVOXEL_LOC {
 public:
  int64_t x, y, z;

  BTCVOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const BTCVOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

namespace std {
template <>
struct hash<BTCVOXEL_LOC> {
  int64_t operator()(const BTCVOXEL_LOC &s) const {
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

class STD_LOC {
 public:
  int64_t x, y, z;

  STD_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const STD_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

namespace std {
template <>
struct hash<STD_LOC> {
  int64_t operator()(const STD_LOC &s) const {
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

class BTCOctoTree {
 public:
  ConfigSetting config_setting_;
  std::vector<Eigen::Vector3d> voxel_points_;
  BTCPlane *plane_ptr_;

  explicit BTCOctoTree(const ConfigSetting &config_setting)
      : config_setting_(config_setting), plane_ptr_(new BTCPlane) {}

  void init_plane();
  void init_octo_tree();

  ~BTCOctoTree() { delete plane_ptr_; }
};

double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2);

bool binary_greater_sort(const BinaryDescriptor &a,
                         const BinaryDescriptor &b);
bool plane_greater_sort(BTCPlane *plane1, BTCPlane *plane2);

class STDescManager {
 public:
  STDescManager() = default;

  ConfigSetting config_setting_;
  unsigned int current_frame_id_ = 0;

  explicit STDescManager(ConfigSetting &config_setting)
      : config_setting_(config_setting), current_frame_id_(0) {}

  // hash table, save all descriptors
  std::unordered_map<STD_LOC, std::vector<STD>> data_base_;

  // save all planes of key frame, required
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> plane_cloud_vec_;
  std::vector<size_t> frame_std_counts_;

  struct VerifiedCandidate {
    int frame_id = -1;
    double score = 0.0;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> transform;
  };
  std::vector<VerifiedCandidate> last_verified_candidates_;

  // generate STDescs from a point cloud
  void GenerateSTDescs(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                       std::vector<STD> &stds_vec, int id);

  // search result <candidate_id, plane icp score>. -1 for no loop
  void SearchLoop(std::vector<STD> &stds_vec,
                  std::pair<int, double> &loop_result,
                  std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
                  pcl::PointCloud<pcl::PointXYZINormal>::Ptr pl_cur);

  // add descriptors to database
  void AddSTDescs(const std::vector<STD> &stds_vec);

 private:
  // voxelization and plane detection
  void init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::unordered_map<BTCVOXEL_LOC, BTCOctoTree *> &voxel_map);

  // acquire planes from voxel_map
  void get_plane(const std::unordered_map<BTCVOXEL_LOC, BTCOctoTree *> &voxel_map,
                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud);

  void get_project_plane(std::unordered_map<BTCVOXEL_LOC, BTCOctoTree *> &feat_map,
                         std::vector<BTCPlane *> &project_plane_list);

  void merge_plane(std::vector<BTCPlane *> &origin_list,
                   std::vector<BTCPlane *> &merge_plane_list);

  // extract corner points from pre-built voxel map and clouds
  void binary_extractor(const std::vector<BTCPlane *> &proj_plane_list,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        std::vector<BinaryDescriptor> &binary_descriptor_list);

  void extract_binary(const Eigen::Vector3d &project_center,
                      const Eigen::Vector3d &project_normal,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::vector<BinaryDescriptor> &binary_list);

  // non maximum suppression, to control the number of corners
  void non_maxi_suppression(std::vector<BinaryDescriptor> &binary_list);

  // build STDescs from corner points.
  void generate_std(const std::vector<BinaryDescriptor> &binary_list,
                    const int &frame_id, std::vector<STD> &std_list);

  // Select a specified number of candidate frames according to the number of
  // STDesc rough matches
  void candidate_selector(std::vector<STD> &stds_vec,
                          std::vector<STDMatchList> &candidate_matcher_vec);

  // Get the best candidate frame by geometry check
  void candidate_verify(
      STDMatchList &candidate_matcher, double &verify_score,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr pl_cur);

  // Get the transform between a matched std pair
  void triangle_solver(std::pair<STD, STD> &std_pair, Eigen::Vector3d &t,
                       Eigen::Matrix3d &rot);

  // Geometrical verification by plane-to-plane icp threshold
  double plane_geometric_verify(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform);
};
