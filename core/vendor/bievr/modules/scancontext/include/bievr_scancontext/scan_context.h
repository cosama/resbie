#ifndef BIEVR_SCANCONTEXT_SCAN_CONTEXT_H_
#define BIEVR_SCANCONTEXT_SCAN_CONTEXT_H_

// Scan Context (Kim & Kim, IROS 18) place descriptor.
//
// Ported from SC-A-LOAM's SCManager. Depends on Eigen only.

#include <Eigen/Dense>

#include <memory>
#include <utility>
#include <vector>

#include "bievr_scancontext/KDTreeVectorOfVectorsAdaptor.h"

namespace bievr {

// Body-frame (sensor-centric) points, 3xN. Descriptors are not comparable
// across different sensor origins.
using ScanContextCloud = Eigen::Ref<const Eigen::Matrix<double, 3, Eigen::Dynamic>>;

class ScanContext {
 public:
  struct Config {
    int num_rings = 20;      // radial bins
    int num_sectors = 60;    // azimuthal bins
    double max_radius = 80.0;  // points beyond this are ignored [m]
    // Added to every point's z so bins hold height above ground: set it to the
    // sensor origin's height above the ground plane. Too small and occupied
    // bins go negative, colliding with the zero that marks an empty bin.
    double lidar_height = 2.0;
    int num_exclude_recent = 30;  // keyframes before the query that cannot close a loop
    int num_candidates = 3;       // kNN candidates pulled from the ring-key tree
    double search_ratio = 0.1;    // fraction of shifts searched around the sector-key guess
    double dist_threshold = 0.2;  // accept a match below this cosine distance
    // Rebuild the ring-key kd-tree every N queries. Above 1, a keyframe can
    // be matched only once a rebuild has picked it up (up to N queries after
    // it left the exclusion window); 1 is exact, larger values save time online.
    int tree_making_period = 1;
  };

  struct Match {
    int index = -1;  // -1 means no match
    float yaw_diff_rad = 0.0f;
    double distance = 1.0;  // best candidate's distance, match or not
  };

  explicit ScanContext(Config config);

  // Appends a keyframe to the database.
  void add(const ScanContextCloud& cloud_body);

  // Matches the newest entry against the older ones, skipping the most recent
  // `num_exclude_recent`.
  Match detectLoop();

  size_t size() const { return descriptors_.size(); }

 private:
  Eigen::MatrixXd makeDescriptor(const ScanContextCloud& cloud) const;

  using KeyMat = std::vector<std::vector<float>>;
  using KeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

  Eigen::MatrixXd ringKey(const Eigen::MatrixXd& desc) const;
  Eigen::MatrixXd sectorKey(const Eigen::MatrixXd& desc) const;

  double directDistance(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const;
  int alignBySectorKey(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const;
  // Best cosine distance over the searched shifts, and the winning shift.
  std::pair<double, int> distance(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const;

  // kNN over `tree` (indexing the first `num_searchable` entries), then exact
  // scoring of the candidates against `desc`.
  Match search(KeyTree& tree, const std::vector<float>& key, const Eigen::MatrixXd& desc,
               size_t num_searchable) const;

  Config config_;
  std::vector<Eigen::MatrixXd> descriptors_;
  KeyMat ring_keys_;

  // detectLoop()'s tree, rebuilt periodically over all but the recent entries.
  KeyMat search_keys_;
  std::unique_ptr<KeyTree> tree_;
  int tree_counter_ = 0;
};

}  // namespace bievr

#endif  // BIEVR_SCANCONTEXT_SCAN_CONTEXT_H_
