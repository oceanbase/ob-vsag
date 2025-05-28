#ifndef OB_VSAG_LIB_H
#define OB_VSAG_LIB_H
#include <stdint.h>
#include <iostream>
#include <map>
namespace obvectorlib {


int64_t example();
typedef void* VectorIndexPtr;
extern bool is_init_;
enum IndexType {
  INVALID_INDEX_TYPE = -1,
  HNSW_TYPE = 0,
  HNSW_SQ_TYPE = 1,
  // Keep it the same as ObVectorIndexAlgorithmType
  // IVF_FLAT_TYPE,
  // IVF_SQ8_TYPE,
  // IVF_PQ_TYPE,
  HNSW_BQ_TYPE = 5,
  HGRAPH_TYPE = 6,
  MAX_INDEX_TYPE
};

class FilterInterface {
public:
  virtual bool test(int64_t id) = 0;
  virtual bool test(const char* data) = 0;
};
/**
 *   * Get the version based on git revision
 *     * 
 *       * @return the version text
 *         */
extern std::string
version();

/**
 *   * Init the vsag library
 *     * 
 *       * @return true always
 *         */
extern bool is_init();

/*
 * *trace = 0
 * *debug = 1
 * *info = 2
 * *warn = 3
 * *err = 4
 * *critical = 5
 * *off = 6
 * */
extern void set_log_level(int32_t ob_level_num);
extern void set_logger(void *logger_ptr);
extern void set_block_size_limit(uint64_t size);
extern bool is_supported_index(IndexType index_type);
extern int create_index(VectorIndexPtr& index_handler, IndexType index_type,
                        const char* dtype,
                        const char* metric,int dim,
                        int max_degree, int ef_construction, int ef_search, void* allocator = NULL,
                        int extra_info_size = 0);
extern int build_index(VectorIndexPtr& index_handler, float* vector_list, int64_t* ids, int dim, int size, char *extra_infos = nullptr);
extern int add_index(VectorIndexPtr& index_handler, float* vector, int64_t* ids, int dim, int size, char *extra_info = nullptr);
extern int get_index_number(VectorIndexPtr& index_handler, int64_t &size);
extern int get_index_type(VectorIndexPtr& index_handler);
extern int cal_distance_by_id(VectorIndexPtr& index_handler, const float* vector, const int64_t* ids, int64_t count, const float *&distances);
extern int get_vid_bound(VectorIndexPtr& index_handler, int64_t &min_vid, int64_t &max_vid);
extern int knn_search(VectorIndexPtr& index_handler,float* query_vector, int dim, int64_t topk,
                      const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
                      bool need_extra_info, const char*& extra_infos,
                      void* invalid, bool reverse_filter, bool use_extra_info_filter,
                      float valid_ratio, void *&iter_ctx, bool is_last_search = false, void *allocator = nullptr);
extern int knn_search(VectorIndexPtr& index_handler,float* query_vector, int dim, int64_t topk,
                      const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
                      bool need_extra_info, const char*& extra_infos,
                      void* invalid = NULL, bool reverse_filter = false,
                      bool use_extra_info_filter = false, void *allocator = nullptr, float valid_ratio = 1);
extern int serialize(VectorIndexPtr& index_handler, const std::string dir);
extern int deserialize_bin(VectorIndexPtr& index_handler, const std::string dir);
extern int fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream);
extern int fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream);
extern int delete_index(VectorIndexPtr& index_handler);
extern void delete_iter_ctx(void *iter_ctx);
extern uint64_t estimate_memory(VectorIndexPtr& index_handler, uint64_t row_count);
extern int get_extra_info_by_ids(VectorIndexPtr& index_handler, 
                                const int64_t* ids, 
                                int64_t count, 
                                char *extra_infos);
} // namesapce obvectorlib
#endif // OB_VSAG_LIB_H

