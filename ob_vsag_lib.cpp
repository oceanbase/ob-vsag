
#include "ob_vsag_lib.h"
#include "ob_vsag_lib_c.h"
#include "nlohmann/json.hpp"
#include "roaring/roaring64.h"
#include <vsag/vsag.h>
#include "vsag/errors.h"
#include "vsag/dataset.h"
#include "vsag/bitset.h"
#include "vsag/allocator.h"
#include "vsag/factory.h"
#include "vsag/search_param.h"
#include "vsag/constants.h"
#include "vsag/filter.h"
#include "vsag/iterator_context.h"

#include "default_logger.h"
#include "vsag/logger.h"

#include <fstream>
#include <chrono>

namespace obvectorlib {

struct SlowTaskTimer {
    SlowTaskTimer(const std::string& name, int64_t log_threshold_ms = 0);
    ~SlowTaskTimer();

    std::string name;
    int64_t threshold;
    std::chrono::steady_clock::time_point start;
};

SlowTaskTimer::SlowTaskTimer(const std::string& n, int64_t log_threshold_ms)
    : name(n), threshold(log_threshold_ms) {
    start = std::chrono::steady_clock::now();
}

SlowTaskTimer::~SlowTaskTimer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    if (duration.count() > threshold) {
        if (duration.count() >= 1000) {
            vsag::logger::debug("  {0} cost {1:.3f}s", name, duration.count() / 1000);
        } else {
            vsag::logger::debug("  {0} cost {1:.3f}ms", name, duration.count());
        }
    }
}

class ObVasgFilter : public vsag::Filter
{
public:
    ObVasgFilter(float valid_ratio,
                 const std::function<bool(int64_t)>& vid_fallback_func,
                 const std::function<bool(const char*)>& exinfo_fallback_func) 
        : valid_ratio_(valid_ratio), vid_fallback_func_(vid_fallback_func), exinfo_fallback_func_(exinfo_fallback_func)
    {};

    ~ObVasgFilter() {}
    
    bool CheckValid(int64_t id) const override {
        return !vid_fallback_func_(id);
    }

    bool CheckValid(const char* data) const override {
        return !exinfo_fallback_func_(data);
    }

    float ValidRatio() const override {
        return valid_ratio_;
    }

private:
    float valid_ratio_;
    std::function<bool(int64_t)> vid_fallback_func_{nullptr};
    std::function<bool(const char*)> exinfo_fallback_func_{nullptr};
};

class HnswIndexHandler
{
public:
  HnswIndexHandler() = delete;

  HnswIndexHandler(bool is_create, bool is_build, bool use_static, const char* dtype, const char* metric, 
                   int max_degree, int ef_construction, int ef_search, int dim, IndexType index_type,
                   std::shared_ptr<vsag::Index> index, vsag::Allocator* allocator, uint64_t extra_info_size):
      is_created_(is_create),
      is_build_(is_build),
      use_static_(use_static),
      dtype_(dtype),
      metric_(metric),
      max_degree_(max_degree),
      ef_construction_(ef_construction),
      ef_search_(ef_search),
      dim_(dim),
      index_type_(index_type),
      index_(index),
      allocator_(allocator),
      extra_info_size_(extra_info_size)
  {}

  ~HnswIndexHandler() {
    index_ = nullptr;
    vsag::logger::debug("   after deconstruction, hnsw index addr {} : use count {}", (void*)allocator_, index_.use_count());
  }
  void set_build(bool is_build) { is_build_ = is_build;}
  bool is_build(bool is_build) { return is_build_;}
  int build_index(const vsag::DatasetPtr& base);
  int get_index_number();
  int add_index(const vsag::DatasetPtr& incremental);
  int cal_distance_by_id(const float* vector, const int64_t* ids, int64_t count, const float*& dist);
  int get_extra_info_by_ids(const int64_t* ids, 
                            int64_t count, 
                            char *extra_infos);
  int get_vid_bound(int64_t &min_vid, int64_t &max_vid);
  uint64_t estimate_memory(uint64_t row_count);
  int knn_search(const vsag::DatasetPtr& query, int64_t topk,
                const std::string& parameters,
                const float*& dist, const int64_t*& ids, int64_t &result_size,
                float valid_ratio, int index_type,
                FilterInterface *bitmap, bool reverse_filter,
                bool need_extra_info, const char*& extra_infos,
                void *allocator);
  int knn_search(const vsag::DatasetPtr& query, int64_t topk,
                const std::string& parameters,
                const float*& dist, const int64_t*& ids, int64_t &result_size,
                float valid_ratio, int index_type,
                FilterInterface *bitmap, bool reverse_filter,
                bool need_extra_info, const char*& extra_infos,
                void *&iter_ctx, bool is_last_search,
                void *allocator);
  std::shared_ptr<vsag::Index>& get_index() {return index_;}
  void set_index(std::shared_ptr<vsag::Index> hnsw) {index_ = hnsw;}
  vsag::Allocator* get_allocator() {return allocator_;}
  inline bool get_use_static() {return use_static_;}
  inline int get_max_degree() {return max_degree_;}
  inline int get_ef_construction() {return ef_construction_;}
  inline int get_index_type() { return (int)index_type_; }
  const char *get_dtype() { return dtype_; }
  const char *get_metric() { return metric_; }
  inline int get_ef_search() {return ef_search_;}
  inline int get_dim() {return dim_;}
  inline uint64_t get_extra_info_size() {return extra_info_size_;}
  
private:
  bool is_created_;
  bool is_build_;
  bool use_static_;
  const char* dtype_; 
  const char* metric_;
  int max_degree_;
  int ef_construction_;
  int ef_search_;
  int dim_;
  IndexType index_type_;
  std::shared_ptr<vsag::Index> index_;
  vsag::Allocator* allocator_;
  uint64_t extra_info_size_;
};

int HnswIndexHandler::build_index(const vsag::DatasetPtr& base) 
{
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    if (const auto num = index_->Build(base); num.has_value()) {
        return 0;
    } else {
        error = num.error().type;
    }
    return static_cast<int>(error);
}

int HnswIndexHandler::get_index_number() 
{
    return index_->GetNumElements();
}

int HnswIndexHandler::add_index(const vsag::DatasetPtr& incremental) 
{
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    if (const auto num = index_->Add(incremental); num.has_value()) {
        vsag::logger::debug(" after add index, index count {}", get_index_number());
        return 0;
    } else {
        error = num.error().type;
    }
    return static_cast<int>(error);
}

int HnswIndexHandler::cal_distance_by_id(const float* vector, const int64_t* ids, int64_t count, const float*& dist)
{
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    auto result = index_->CalDistanceById(vector, ids, count);
    if (result.has_value()) {
        result.value()->Owner(false);
        dist = result.value()->GetDistances();
        return 0;
    } else {
        error = result.error().type;
    }
    return static_cast<int>(error);
}

int HnswIndexHandler::get_extra_info_by_ids(const int64_t* ids, 
                                            int64_t count, 
                                            char *extra_infos)
{
    index_->GetExtraInfoByIds(ids, count, extra_infos);
    return 0;
}

int HnswIndexHandler::get_vid_bound(int64_t &min_vid, int64_t &max_vid)
{
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int64_t element_cnt = index_->GetNumElements();
    if (element_cnt == 0) {
        return 0;
    } else {
        auto result = index_->GetMinAndMaxId();
        if (result.has_value()) {
            min_vid = result.value().first;
            max_vid = result.value().second;
            return 0;
        } else {
            error = result.error().type;
        }
    }
    return static_cast<int>(error);
}

uint64_t HnswIndexHandler::estimate_memory(uint64_t row_count)
{
    return index_->EstimateMemory(row_count);
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr& query, int64_t topk,
               const std::string& parameters,
               const float*& dist, const int64_t*& ids, int64_t &result_size,
               float valid_ratio, int index_type,
               FilterInterface *bitmap, bool reverse_filter,
               bool need_extra_info, const char*& extra_infos, void *allocator) {
    vsag::logger::debug("  search_parameters:{}", parameters);
    vsag::logger::debug("  topk:{}", topk);
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    auto vid_filter = [bitmap, reverse_filter](int64_t id) -> bool {
        if (!reverse_filter) {
            return bitmap->test(id);
        } else {
            return !(bitmap->test(id));
        }
    };
    auto exinfo_filter = [bitmap, reverse_filter](const char* data) -> bool {
        if (!reverse_filter) {
            return bitmap->test(data);
        } else {
            return !(bitmap->test(data));
        }
    };
    tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result;
    auto vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, vid_filter, exinfo_filter);
    vsag::Allocator* vsag_allocator = nullptr;
    if (allocator != nullptr) vsag_allocator =  static_cast<vsag::Allocator*>(allocator);
    vsag::SearchParam search_param(false, parameters, bitmap == nullptr ? nullptr : vsag_filter, vsag_allocator);
    result = index_->KnnSearch(query, topk, search_param);
    if (result.has_value()) {
        //result的生命周期
        result.value()->Owner(false);
        ids = result.value()->GetIds();
        dist = result.value()->GetDistances();
        result_size = result.value()->GetDim();
        if (need_extra_info) {
            extra_infos = result.value()->GetExtraInfos();
        }
        return 0; 
    } else {
        error = result.error().type;
    }

    return static_cast<int>(error);
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr& query, int64_t topk,
               const std::string& parameters,
               const float*& dist, const int64_t*& ids, int64_t &result_size,
               float valid_ratio, int index_type,
               FilterInterface *bitmap, bool reverse_filter,
               bool need_extra_info, const char*& extra_infos,
               void *&iter_ctx, bool is_last_search, void *allocator) {
    vsag::logger::debug("  search_parameters:{}", parameters);
    vsag::logger::debug("  topk:{}", topk);
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    auto filter = [bitmap, reverse_filter](int64_t id) -> bool {
        if (!reverse_filter) {
            return bitmap->test(id);
        } else {
            return !(bitmap->test(id));
        }
    };
    auto exinfo_filter = [bitmap, reverse_filter](const char* data) -> bool {
        if (!reverse_filter) {
            return bitmap->test(data);
        } else {
            return !(bitmap->test(data));
        }
    };
    tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result;
    auto vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, filter, exinfo_filter);
    vsag::Allocator* vsag_allocator = nullptr;
    if (allocator != nullptr) vsag_allocator =  static_cast<vsag::Allocator*>(allocator);
    vsag::IteratorContext* input_iter = static_cast<vsag::IteratorContext*>(iter_ctx);
    vsag::SearchParam search_param(true, parameters, bitmap == nullptr ? nullptr : vsag_filter, vsag_allocator, input_iter, is_last_search);
    result = index_->KnnSearch(query, topk, search_param);
    if (result.has_value()) {
        iter_ctx = input_iter;
        result.value()->Owner(false);
        ids = result.value()->GetIds();
        dist = result.value()->GetDistances();
        result_size = result.value()->GetDim();
        if (need_extra_info) {
            extra_infos = result.value()->GetExtraInfos();
        }
        return 0; 
    } else {
        error = result.error().type;
    }

    return static_cast<int>(error);
}

bool is_init_ = vsag::init();

void
set_log_level(int32_t ob_level_num) {
    std::map<int32_t, int32_t> ob2vsag_log_level = {
        {0 /*ERROR*/, vsag::Logger::Level::kERR},
        {1 /*WARN*/, vsag::Logger::Level::kWARN},
        {2 /*INFO*/, vsag::Logger::Level::kINFO},
        {3 /*EDIAG*/, vsag::Logger::Level::kERR},
        {4 /*WDIAG*/, vsag::Logger::Level::kWARN},
        {5 /*TRACE*/, vsag::Logger::Level::kTRACE},
        {6 /*DEBUG*/, vsag::Logger::Level::kDEBUG},
    };
    vsag::Options::Instance().logger()->SetLevel(
        static_cast<vsag::Logger::Level>(ob2vsag_log_level[ob_level_num]));
}

bool is_init() {
    vsag::logger::debug("TRACE LOG[Init VsagLib]:");
    if (is_init_) {
        vsag::logger::debug("   Init VsagLib success");
    } else {
        vsag::logger::debug("   Init VsagLib fail");
    }
    return is_init_; 
}


void set_logger(void *logger_ptr) {
    vsag::Options::Instance().set_logger(static_cast<vsag::Logger*>(logger_ptr));
    vsag::Logger::Level log_level = static_cast<vsag::Logger::Level>(1);//default is debug level
    vsag::Options::Instance().logger()->SetLevel(log_level);
}

void set_block_size_limit(uint64_t size) {
    vsag::Options::Instance().set_block_size_limit(size);
}

bool is_supported_index(IndexType index_type) {
    return INVALID_INDEX_TYPE < index_type && index_type < MAX_INDEX_TYPE;
}

int create_index(VectorIndexPtr& index_handler, IndexType index_type,
                 const char* dtype,
                 const char* metric, int dim,
                 int max_degree, int ef_construction, int ef_search, void* allocator,
                 int extra_info_size/* = 0*/)
{   
    vsag::logger::debug("TRACE LOG[create_index]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (dtype == nullptr || metric == nullptr) {
        vsag::logger::debug("   null pointer addr, dtype:{}, metric:{}", (void*)dtype, (void*)metric);
        return static_cast<int>(vsag::ErrorType::UNKNOWN_ERROR);
    }
    // SlowTaskTimer t("z");
    vsag::Allocator* vsag_allocator = NULL;
    bool is_support = is_supported_index(index_type);
    vsag::logger::debug("   index type : {}, is_supported : {}", static_cast<int>(index_type), is_support);
    if (allocator == NULL) {
        vsag_allocator = NULL;
        vsag::logger::debug("   allocator is null ,use default_allocator");
    } else {
        vsag_allocator =  static_cast<vsag::Allocator*>(allocator);
        vsag::logger::debug("   allocator_addr:{}",allocator);
    }
    nlohmann::json index_parameters;
    std::string index_type_str;

    if (index_type == HNSW_TYPE) { // {"dim":128,"dtype":"abc","hnsw":{"ef_construction":200,"ef_search":200,"max_degree":16,"use_static":false},"metric_type":"l2"}
        // create index
        bool use_static = false;
        index_type_str = "hnsw";
        nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                {"ef_construction", ef_construction},
                                {"ef_search", ef_search},
                                {"use_static", use_static}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"hnsw", hnsw_parameters}}; 
    } else if (index_type == HNSW_SQ_TYPE) {
        // create hnsw sq index
        index_type_str = "hgraph";
        max_degree *= 2;
        nlohmann::json hnswsq_parameters{{"base_quantization_type", "sq8"},
                                         // NOTE(liyao): max_degree compatible with behavior of HNSW, which is doubling the m value 
                                         {"max_degree", max_degree}, 
                                         {"ef_construction", ef_construction},
                                         {"build_thread_count", 0}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"extra_info_size", extra_info_size}, {"index_param", hnswsq_parameters}}; 
    } else if (index_type == HGRAPH_TYPE) {
        // create hnsw fp index
        index_type_str = "hgraph";
        max_degree *= 2;
        nlohmann::json hnswsq_parameters{{"base_quantization_type", "fp32"},
                                         // NOTE(liyao): max_degree compatible with behavior of HNSW, which is doubling the m value 
                                         {"max_degree", max_degree}, 
                                         {"ef_construction", ef_construction},
                                         {"build_thread_count", 0}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"extra_info_size", extra_info_size}, {"index_param", hnswsq_parameters}}; 
    } else if (index_type == HNSW_BQ_TYPE) {
        // create hnsw bq index
        index_type_str = "hgraph";
        max_degree *= 2;
        nlohmann::json hnswsq_parameters{{"base_quantization_type", "rabitq"},
                                         // NOTE(liyao): max_degree compatible with behavior of HNSW, which is doubling the m value 
                                         {"max_degree", max_degree}, 
                                         {"ef_construction", ef_construction},
                                         {"build_thread_count", 0},
                                         {"use_reorder", true},
                                         {"ignore_reorder", true},
                                         {"precise_quantization_type", "fp32"},
                                         {"precise_io_type", "block_memory_io"}}; 
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"extra_info_size", extra_info_size}, {"index_param", hnswsq_parameters}}; 
    } else if (!is_support) {
        error = vsag::ErrorType::UNSUPPORTED_INDEX;
        vsag::logger::debug("   fail to create hnsw index , index type not supported:{}", static_cast<int>(index_type));
        return static_cast<int>(error);
    }

    if (auto index = vsag::Factory::CreateIndex(index_type_str, index_parameters.dump(), vsag_allocator);
        index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler* hnsw_index = new HnswIndexHandler(true,
                                                            false,
                                                            false,
                                                            dtype,
                                                            metric,
                                                            max_degree,
                                                            ef_construction,
                                                            ef_search,
                                                            dim,
                                                            index_type,
                                                            hnsw,
                                                            vsag_allocator,
                                                            extra_info_size);
        index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        vsag::logger::debug("   success to create hnsw index , index parameter:{}, allocator addr:{}",index_parameters.dump(), (void*)vsag_allocator);
        return 0;
    } else {
        error = index.error().type;
        vsag::logger::debug("   fail to create hnsw index , index parameter:{}", index_parameters.dump());
    }
    ret = static_cast<int>(error);
    if (ret != 0) {
        vsag::logger::error("   create index error happend, ret={}", static_cast<int>(error));
    }
    return ret;
}

int build_index(VectorIndexPtr& index_handler, float* vector_list, int64_t* ids, int dim, int size, char *extra_infos/* = nullptr*/) {
    vsag::logger::debug("TRACE LOG[build_index]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret =  0;
    if (index_handler == nullptr || vector_list == nullptr || ids == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}, ids:{}, ids:{}",
                                                   (void*)index_handler, (void*)vector_list, (void*)ids);
        return static_cast<int>(error);
    }
    // SlowTaskTimer t("build_index");
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)
           ->NumElements(size)
           ->Ids(ids)
           ->Float32Vectors(vector_list)
           ->Owner(false);
    if (extra_infos != nullptr) {
        dataset->ExtraInfos(extra_infos);
    }
    ret = hnsw->build_index(dataset);
    if (ret != 0) {
        vsag::logger::error("   build index error happend, ret={}", ret);
    }
    return ret;
}


int add_index(VectorIndexPtr& index_handler, float* vector, int64_t* ids, int dim, int size, char *extra_info/* = nullptr*/) {
    vsag::logger::debug("TRACE LOG[add_index]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr || vector == nullptr || ids == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}, ids:{}, ids:{}",
                                                   (void*)index_handler, (void*)vector, (void*)ids);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    // SlowTaskTimer t("add_index");
    // add index
    auto incremental = vsag::Dataset::Make();
    incremental->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector)
        ->Owner(false);
    if (extra_info != nullptr) {
        incremental->ExtraInfos(extra_info);
    }
    ret = hnsw->add_index(incremental);
    if (ret != 0) {
        vsag::logger::error("   add index error happend, ret={}", ret);
    }
    return ret;
}

int get_index_type(VectorIndexPtr& index_handler) {
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    return hnsw->get_index_type(); 
}

int get_index_number(VectorIndexPtr& index_handler, int64_t &size) {
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    size = hnsw->get_index_number();
    return 0;
}

int cal_distance_by_id(VectorIndexPtr& index_handler, 
                        const float* vector, 
                        const int64_t* ids, 
                        int64_t count, 
                        const float *&distances)
{
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler); 
    int ret = hnsw->cal_distance_by_id(vector, ids, count, distances);
    if (ret != 0) {
        vsag::logger::error("   knn search error happend, ret={}", ret);
    }
    return ret;
}

extern int get_vid_bound(VectorIndexPtr& index_handler, int64_t &min_vid, int64_t &max_vid)
{
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler); 
    int ret = hnsw->get_vid_bound(min_vid, max_vid);
    if (ret != 0) {
        vsag::logger::error("   get vid bound error happend, ret={}", ret);
    }
    return ret;
}

int knn_search(VectorIndexPtr& index_handler, float* query_vector,int dim, int64_t topk,
               const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
               bool need_extra_info, const char*& extra_infos,
               void* invalid, bool reverse_filter, bool use_extra_info_filter, void *allocator,
               float valid_ratio) {
    vsag::logger::debug("TRACE LOG[knn_search]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr || query_vector == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}, query_vector:{}",
                                                   (void*)index_handler, (void*)query_vector);
        return static_cast<int>(error);
    }
    // SlowTaskTimer t("knn_search");
    FilterInterface *bitmap = static_cast<FilterInterface*>(invalid);
    bool owner_set = false;
    nlohmann::json search_parameters;
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    const IndexType index_type =static_cast<IndexType>(hnsw->get_index_type());
    if (HNSW_SQ_TYPE == index_type || HNSW_BQ_TYPE == index_type || HGRAPH_TYPE == index_type) {
        search_parameters = {{"hgraph", {{"ef_search", ef_search}, {"use_extra_info_filter", use_extra_info_filter}}},};
        owner_set = true;
    } else {
        search_parameters = {{"hnsw", {{"ef_search", ef_search}, {"skip_ratio", 0.7f}}},};
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
    ret = hnsw->knn_search(
        query, topk, search_parameters.dump(), dist, ids, result_size, valid_ratio, index_type,
        bitmap, reverse_filter,
        need_extra_info, extra_infos, allocator);
    if (ret != 0) {
        vsag::logger::error("   knn search error happend, ret={}", ret);
    }
    return ret;
}

int knn_search(VectorIndexPtr& index_handler, float* query_vector,int dim, int64_t topk,
               const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
               bool need_extra_info, const char*& extra_infos,
               void* invalid, bool reverse_filter, bool use_extra_info_filter,
               float valid_ratio, void *&iter_ctx, bool is_last_search, void *allocator) {
    vsag::logger::debug("TRACE LOG[knn_search]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr || query_vector == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}, query_vector:{}",
                                                   (void*)index_handler, (void*)query_vector);
        return static_cast<int>(error);
    }
    // SlowTaskTimer t("knn_search");
    FilterInterface *bitmap = static_cast<FilterInterface*>(invalid);
    bool owner_set = false;
    nlohmann::json search_parameters;
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    const IndexType index_type =static_cast<IndexType>(hnsw->get_index_type());
    if (HNSW_SQ_TYPE == index_type || HNSW_BQ_TYPE == index_type || HGRAPH_TYPE == index_type) { // {"hgraph":{"ef_search":200,"use_extra_info_filter":true}}
        search_parameters = {{"hgraph", {{"ef_search", ef_search}, {"use_extra_info_filter", use_extra_info_filter}}},};
        owner_set = true;
    } else {
        search_parameters = {{"hnsw", {{"ef_search", ef_search}, {"skip_ratio", 0.7f}}},};
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
    ret = hnsw->knn_search(
        query, topk, search_parameters.dump(), dist, ids, result_size, valid_ratio, index_type,
        bitmap, reverse_filter,
        need_extra_info, extra_infos, 
        iter_ctx, is_last_search, allocator);
    if (ret != 0) {
        vsag::logger::error("   knn search error happend, ret={}", ret);
    }
    return ret;
}

int serialize(VectorIndexPtr& index_handler, const std::string dir) {
    vsag::logger::debug("TRACE LOG[serialize]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret =  0;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    if (auto bs = hnsw->get_index()->Serialize(); bs.has_value()) {
        hnsw = nullptr;
        auto keys = bs->GetKeys();
        for (auto key : keys) {
            vsag::Binary b = bs->Get(key);
            std::ofstream file(dir + "hnsw.index." + key, std::ios::binary);
            file.write((const char*)b.data.get(), b.size);
            file.close();
        }
        std::ofstream metafile(dir + "hnsw.index._meta", std::ios::out);
        for (auto key : keys) {
            metafile << key << std::endl;
        }
        metafile.close();
        return 0;
    } else {
        error = bs.error().type;
    }
    ret = static_cast<int>(error);
    if (ret != 0) {
        vsag::logger::error("   serialize error happend, ret={}", ret);
    }
    return ret;
}

int fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream) {
    vsag::logger::debug("TRACE LOG[fserialize]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    if (auto bs = hnsw->get_index()->Serialize(out_stream); bs.has_value()) {
        return 0;
    } else {
        error = bs.error().type;
    }
    ret = static_cast<int>(error);
    if (ret != 0) {
        vsag::logger::error("   fserialize error happend, ret={}", ret);
    }
    return ret;
}

int fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream) {
    vsag::logger::debug("TRACE LOG[fdeserialize]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    std::shared_ptr<vsag::Index> hnsw_index;
    bool use_static = hnsw->get_use_static();
    const char *metric = hnsw->get_metric();
    const char *dtype = hnsw->get_dtype();
    int max_degree = hnsw->get_max_degree();
    int ef_construction = hnsw->get_ef_construction();
    int ef_search = hnsw->get_ef_search();
    int dim = hnsw->get_dim();
    int index_type = hnsw->get_index_type();
    uint64_t extra_info_size = hnsw->get_extra_info_size();
    const char *base_quantization_type = (index_type == HNSW_SQ_TYPE) ? "sq8" : ((index_type == HNSW_BQ_TYPE) ? "rabitq" : "fp32");
    nlohmann::json index_parameters;
    if (HNSW_TYPE == index_type) {
        nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                    {"ef_construction", ef_construction},
                                    {"ef_search", ef_search},
                                    {"use_static", use_static}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    } else if (HNSW_SQ_TYPE == index_type) {
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                        {"max_degree", max_degree},
                                        {"ef_construction", ef_construction},
                                        {"build_thread_count", 0}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"extra_info_size", extra_info_size}, {"index_param", hnswsq_parameters}};
    } else if (HNSW_BQ_TYPE == index_type) {
        nlohmann::json hnswbq_parameters{{"base_quantization_type", "rabitq"}, 
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 0},
                                            {"use_reorder", true},
                                            {"ignore_reorder", true},
                                            {"precise_quantization_type", "fp32"},
                                            {"precise_io_type", "block_memory_io"}};  
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"extra_info_size", extra_info_size}, {"index_param", hnswbq_parameters}};
    } else if (HGRAPH_TYPE == index_type) {
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                        {"max_degree", max_degree},
                                        {"ef_construction", ef_construction},
                                        {"build_thread_count", 0}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"extra_info_size", extra_info_size}, {"index_param", hnswsq_parameters}};       
    }

    vsag::logger::debug("   Deserilize hnsw index , index parameter:{}, allocator addr:{}",index_parameters.dump(),(void*)hnsw->get_allocator());
    if (index_type == HNSW_TYPE) {
        if (auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump(), hnsw->get_allocator());
            index.has_value()) {
            hnsw_index = index.value();
        } else {
            error = index.error().type;
            return static_cast<int>(error);
        }
    } else {
        if (auto index = vsag::Factory::CreateIndex("hgraph", index_parameters.dump(), hnsw->get_allocator());
            index.has_value()) {
            hnsw_index = index.value();
        } else {
            error = index.error().type;
            return static_cast<int>(error);
        }        
    }
    if (ret != 0) {

    } else if (auto bs = hnsw_index->Deserialize(in_stream); bs.has_value()) {
        hnsw->set_index(hnsw_index);
        return 0;
    } else {
        error = bs.error().type;
        ret = static_cast<int>(error);
    }
    if (ret != 0) {
        vsag::logger::error("   fdeserialize error happend, ret={}", ret);
    }
    return ret;
}

int deserialize_bin(VectorIndexPtr& index_handler,const std::string dir) {
    vsag::logger::debug("TRACE LOG[deserialize]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler={}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    std::ifstream metafile(dir + "hnsw.index._meta", std::ios::in);
    std::vector<std::string> keys;
    std::string line;
    while (std::getline(metafile, line)) {
        keys.push_back(line);
    }
    metafile.close();

    vsag::BinarySet bs;
    for (auto key : keys) {
        std::ifstream file(dir + "hnsw.index." + key, std::ios::in);
        file.seekg(0, std::ios::end);
        vsag::Binary b;
        b.size = file.tellg();
        b.data.reset(new int8_t[b.size]);
        file.seekg(0, std::ios::beg);
        file.read((char*)b.data.get(), b.size);
        bs.Set(key, b);
    }
    bool use_static = hnsw->get_use_static();
    const char *metric = hnsw->get_metric();
    const char *dtype = hnsw->get_dtype();
    int max_degree = hnsw->get_max_degree();
    int ef_construction = hnsw->get_ef_construction();
    int ef_search = hnsw->get_ef_search();
    int dim = hnsw->get_dim();
    int index_type = hnsw->get_index_type();
    uint64_t extra_info_size = hnsw->get_extra_info_size();
    const char *base_quantization_type = (index_type == HNSW_SQ_TYPE) ? "sq8" : ((index_type == HNSW_BQ_TYPE) ? "rabitq" : "fp32");
    nlohmann::json index_parameters;
    if (index_type == HNSW_TYPE) {
        nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                    {"ef_construction", ef_construction},
                                    {"ef_search", ef_search},
                                    {"use_static", use_static}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"hnsw", hnsw_parameters}};
     } else if (index_type == HNSW_BQ_TYPE) {
        nlohmann::json hnswbq_parameters{{"base_quantization_type", base_quantization_type}, 
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 0},
                                            {"extra_info_size", extra_info_size},
                                            {"use_reorder", true},
                                            {"ignore_reorder", true},
                                            {"precise_quantization_type", "fp32"},
                                            {"precise_io_type", "block_memory_io"}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswbq_parameters}};
    } else {
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 0},
                                            {"extra_info_size", extra_info_size}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswsq_parameters}};
    }
   
    vsag::logger::debug("   Deserilize hnsw index , index parameter:{}, allocator addr:{}",index_parameters.dump(),(void*)hnsw->get_allocator());
    std::shared_ptr<vsag::Index> hnsw_index;
    if (index_type == HNSW_TYPE) {
        if (auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump(), hnsw->get_allocator());
            index.has_value()) {
            hnsw_index = index.value();
        } else {
            error = index.error().type;
            return static_cast<int>(error);
        }
    } else {
        if (auto index = vsag::Factory::CreateIndex("hgraph", index_parameters.dump(), hnsw->get_allocator());
            index.has_value()) {
            hnsw_index = index.value();
        } else {
            error = index.error().type;
            return static_cast<int>(error);
        }        
    }
    hnsw_index->Deserialize(bs);
    hnsw->set_index(hnsw_index);
    return 0;
}

int delete_index(VectorIndexPtr& index_handler) {
    vsag::logger::debug("TRACE LOG[delete_index]");
    vsag::logger::debug("   delete index handler addr {} : hnsw index use count {}",(void*)static_cast<HnswIndexHandler*>(index_handler)->get_index().get(),static_cast<HnswIndexHandler*>(index_handler)->get_index().use_count());
    if (index_handler != NULL) {
        delete static_cast<HnswIndexHandler*>(index_handler);
        index_handler = NULL;
    }
    return 0;
}

void delete_iter_ctx(void *iter_ctx) {
    vsag::logger::debug("TRACE LOG[delete_iter_ctx]");
    if (iter_ctx != NULL) {
        delete static_cast<vsag::IteratorContext*>(iter_ctx);
        iter_ctx = NULL;
    }
}

int get_extra_info_by_ids(VectorIndexPtr& index_handler, 
                          const int64_t* ids, 
                          int64_t count, 
                          char *extra_infos) {
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    if (index_handler == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}", (void*)index_handler);
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler); 
    int ret = hnsw->get_extra_info_by_ids(ids, count, extra_infos);
    if (ret != 0) {
        vsag::logger::error("   knn search error happend, ret={}", ret);
    }
    return ret;
}

uint64_t estimate_memory(VectorIndexPtr& index_handler,
                         uint64_t row_count) {
    vsag::logger::debug("TRACE LOG[estimate_memory]");
    uint64_t estimate_memory_size = 0;
    if (index_handler != nullptr) {
        HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler); 
        estimate_memory_size = hnsw->estimate_memory(row_count);
    }
    return estimate_memory_size;
}

int64_t example() {
    return 0;
}

extern bool is_init_c() {
    return is_init();
}

extern void set_logger_c(void *logger_ptr) {
    set_logger(logger_ptr);
}

extern void set_block_size_limit_c(uint64_t size) {
    set_block_size_limit(size);
}

extern bool is_supported_index_c(IndexType index_type) {
    return is_supported_index(index_type);
}

extern int create_index_c(VectorIndexPtr& index_handler, IndexType index_type,
                 const char* dtype,
                 const char* metric, int dim,
                 int max_degree, int ef_construction, int ef_search, void* allocator)
{   
    return create_index(index_handler, index_type, dtype, metric, dim, max_degree, ef_construction, ef_search, allocator);
}

extern int build_index_c(VectorIndexPtr& index_handler,float* vector_list, int64_t* ids, int dim, int size) {

    return build_index(index_handler, vector_list, ids, dim, size);
}


extern int add_index_c(VectorIndexPtr& index_handler,float* vector, int64_t* ids, int dim, int size) {
    return  add_index(index_handler, vector, ids, dim, size);
}

extern int get_index_number_c(VectorIndexPtr& index_handler, int64_t &size) {
    return get_index_number(index_handler, size);
}

extern int get_index_type_c(VectorIndexPtr& index_handler) {
    return get_index_type(index_handler);
}

extern int knn_search_c(VectorIndexPtr& index_handler,float* query_vector,int dim, int64_t topk,
               const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search, 
               bool need_extra_info, const char*& extra_infos,
               void* invalid, bool reverse_filter, bool use_extra_info_filter) {
    return knn_search(index_handler, query_vector, dim, topk, dist, ids, result_size,
                      ef_search, need_extra_info, extra_infos, invalid, reverse_filter, use_extra_info_filter);
}

extern int serialize_c(VectorIndexPtr& index_handler, const std::string dir) {
    return serialize(index_handler, dir);
}

extern int fserialize_c(VectorIndexPtr& index_handler, std::ostream& out_stream) {
    return fserialize(index_handler, out_stream);
}

extern int delete_index_c(VectorIndexPtr& index_handler) {
    return delete_index(index_handler);
}
extern int fdeserialize_c(VectorIndexPtr& index_handler, std::istream& in_stream) {
    return fdeserialize(index_handler, in_stream);
}

extern int deserialize_bin_c(VectorIndexPtr& index_handler,const std::string dir) {
    return deserialize_bin(index_handler, dir);
}

extern uint64_t estimate_memory_c(VectorIndexPtr& index_handler, uint64_t row_count) {
    return estimate_memory(index_handler, row_count);
}

} //namespace obvectorlib
