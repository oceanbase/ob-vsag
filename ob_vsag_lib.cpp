
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
    ObVasgFilter(float valid_ratio, const std::function<bool(int64_t)>& fallback_func) 
        : valid_ratio_(valid_ratio), fallback_func_(fallback_func) 
    {};

    ~ObVasgFilter() {}
    
    bool CheckValid(int64_t id) const override {
        return !fallback_func_(id);
    }

    float ValidRatio() const override {
        return valid_ratio_;
    }

private:
    float valid_ratio_;
    std::function<bool(int64_t)> fallback_func_{nullptr};
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
  int get_vid_bound(int64_t &min_vid, int64_t &max_vid);
  int knn_search(const vsag::DatasetPtr& query, int64_t topk,
                const std::string& parameters,
                const float*& dist, const int64_t*& ids, int64_t &result_size,
                float valid_ratio, int index_type,
                FilterInterface *bitmap, bool reverse_filter,
                bool need_extra_info, const char*& extra_infos,
                void *iter_ctx = nullptr, bool is_last_search = false);
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

int HnswIndexHandler::get_vid_bound(int64_t &min_vid, int64_t &max_vid)
{
    auto result = index_->GetMinAndMaxId(min_vid, max_vid);
    return 0;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr& query, int64_t topk,
               const std::string& parameters,
               const float*& dist, const int64_t*& ids, int64_t &result_size,
               float valid_ratio, int index_type,
               FilterInterface *bitmap, bool reverse_filter,
               bool need_extra_info, const char*& extra_infos,
               void *iter_ctx, bool is_last_search) {
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
    tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result;
    auto vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, filter);
    if (iter_ctx == nullptr) { // knn search without iter filter
        result = (index_type == HNSW_TYPE || index_type == HGRAPH_TYPE) ?
                    index_->KnnSearch(query, topk, parameters, bitmap == nullptr ? nullptr : vsag_filter) :
                    index_->KnnSearch(query, topk, parameters, filter);
    } else { // iter filter
        vsag::IteratorContextPtr* input_iter = static_cast<vsag::IteratorContextPtr*>(iter_ctx);
        if (index_type == HNSW_TYPE || index_type == HGRAPH_TYPE || index_type == HNSW_SQ_TYPE) {
            result = index_->KnnSearch(query, topk, parameters, bitmap == nullptr ? nullptr : vsag_filter, input_iter, is_last_search);
        } else {
            error = vsag::ErrorType::UNSUPPORTED_INDEX;
            vsag::logger::error("knn search iter filter not support BQ.");
        }
    }
    if (result.has_value()) {
        //result的生命周期
        result.value()->Owner(false);
        ids = result.value()->GetIds();
        dist = result.value()->GetDistances();
        result_size = result.value()->GetDim();
        if (need_extra_info) {
            extra_infos = result.value()->GetExtraInfos();
        }
        // print the results
        for (int64_t i = 0; i < result_size; ++i) {
            vsag::logger::debug("  knn search id : {}, distance : {}",ids[i],dist[i]);
        }
        return 0; 
    } else {
        error = result.error().type;
    }

    return static_cast<int>(error);
}

bool is_init_ = vsag::init();

void
set_log_level(int64_t level_num) {
    vsag::Logger::Level log_level = static_cast<vsag::Logger::Level>(level_num);
    vsag::Options::Instance().logger()->SetLevel(log_level);
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
        return static_cast<int>(error);
    }
    SlowTaskTimer t("z");
    vsag::Allocator* vsag_allocator = NULL;
    bool is_support = is_supported_index(index_type);
    const char *base_quantization_type = (index_type == HNSW_SQ_TYPE) ? "sq8" : ((index_type == HNSW_BQ_TYPE) ? "rabitq" : "fp32");
    vsag::logger::debug("   index type : {}, is_supported : {}", static_cast<int>(index_type), is_support);
    if (allocator == NULL) {
        vsag_allocator = NULL;
        vsag::logger::debug("   allocator is null ,use default_allocator");
    } else {
        vsag_allocator =  static_cast<vsag::Allocator*>(allocator);
        vsag::logger::debug("   allocator_addr:{}",allocator);
    }

    if (index_type == HNSW_TYPE) {
        // create index
        std::shared_ptr<vsag::Index> hnsw;
        bool use_static = false;
        nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                {"ef_construction", ef_construction},
                                {"ef_search", ef_search},
                                {"use_static", use_static}};
        nlohmann::json index_parameters{{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"hnsw", hnsw_parameters}}; 
        if (auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump(), vsag_allocator);
            index.has_value()) {
            hnsw = index.value();
            HnswIndexHandler* hnsw_index = new HnswIndexHandler(true,
                                                                false,
                                                                use_static,
                                                                dtype,
                                                                metric,
                                                                max_degree,
                                                                ef_construction,
                                                                ef_search,
                                                                dim,
                                                                index_type,
                                                                hnsw,
                                                                vsag_allocator,
                                                                0/*extra_info_size*/);
            index_handler = static_cast<VectorIndexPtr>(hnsw_index);
            vsag::logger::debug("   success to create hnsw index , index parameter:{}, allocator addr:{}",index_parameters.dump(), (void*)vsag_allocator);
            return 0;
        } else {
            error = index.error().type;
            vsag::logger::debug("   fail to create hnsw index , index parameter:{}", index_parameters.dump());
        }
    } else if (index_type == HNSW_SQ_TYPE || index_type == HGRAPH_TYPE) {
        // create hnsw sq index
        std::shared_ptr<vsag::Index> hnsw;
        bool use_static = false;
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 1},
                                            {"extra_info_size", extra_info_size}};
        nlohmann::json index_parameters{{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswsq_parameters}}; 
        if (auto index = vsag::Factory::CreateIndex("hgraph", index_parameters.dump(), vsag_allocator);
            index.has_value()) {
            hnsw = index.value();
            HnswIndexHandler* hnsw_index = new HnswIndexHandler(true,
                                                                false,
                                                                use_static,
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
    } else if (index_type == HNSW_BQ_TYPE) {
        // create hnsw sq index
        std::shared_ptr<vsag::Index> hnsw;
        bool use_static = false;
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 1},
                                            {"extra_info_size", extra_info_size},
                                            {"use_reorder", true},
                                            {"precise_quantization_type", "fp32"},
                                            {"precise_io_type", "block_memory_io"}}; 
        nlohmann::json index_parameters{{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswsq_parameters}}; 
        if (auto index = vsag::Factory::CreateIndex("hgraph", index_parameters.dump(), vsag_allocator);
            index.has_value()) {
            hnsw = index.value();
            HnswIndexHandler* hnsw_index = new HnswIndexHandler(true,
                                                                false,
                                                                use_static,
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
    } else if (!is_support) {
        error = vsag::ErrorType::UNSUPPORTED_INDEX;
        vsag::logger::debug("   fail to create hnsw index , index type not supported:{}", static_cast<int>(index_type));
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
    SlowTaskTimer t("build_index");
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
    SlowTaskTimer t("add_index");
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
        vsag::logger::error("   knn search error happend, ret={}", ret);
    }
    return ret;
}

int knn_search(VectorIndexPtr& index_handler, float* query_vector,int dim, int64_t topk,
               const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
               bool need_extra_info, const char*& extra_infos,
               void* invalid, bool reverse_filter, float valid_ratio, 
               void *iter_ctx, bool is_last_search) {
    vsag::logger::debug("TRACE LOG[knn_search]:");
    vsag::ErrorType error = vsag::ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (index_handler == nullptr || query_vector == nullptr) {
        vsag::logger::debug("   null pointer addr, index_handler:{}, query_vector:{}",
                                                   (void*)index_handler, (void*)query_vector);
        return static_cast<int>(error);
    }
    SlowTaskTimer t("knn_search");
    FilterInterface *bitmap = static_cast<FilterInterface*>(invalid);
    bool owner_set = false;
    nlohmann::json search_parameters;
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    const IndexType index_type =static_cast<IndexType>(hnsw->get_index_type());
    if (HNSW_SQ_TYPE == index_type || HNSW_BQ_TYPE == index_type || HGRAPH_TYPE == index_type) {
        search_parameters = {{"hgraph", {{"ef_search", ef_search}}},};
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
        iter_ctx, is_last_search);
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
    } else if (HNSW_BQ_TYPE == index_type) {
        nlohmann::json hnswbq_parameters{{"base_quantization_type", "rabitq"}, 
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 1},
                                            {"extra_info_size", extra_info_size},
                                            {"use_reorder", true},
                                            {"precise_quantization_type", "fp32"},
                                            {"precise_io_type", "block_memory_io"}};  
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswbq_parameters}};
    } else {
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                        {"max_degree", max_degree},
                                        {"ef_construction", ef_construction},
                                        {"build_thread_count", 1},
                                        {"extra_info_size", extra_info_size}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswsq_parameters}};
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
                                            {"build_thread_count", 1},
                                            {"extra_info_size", extra_info_size},
                                            {"use_reorder", true},
                                            {"precise_quantization_type", "fp32"},
                                            {"precise_io_type", "block_memory_io"}};
        index_parameters = {{"dtype", dtype}, {"metric_type", metric}, {"dim", dim}, {"index_param", hnswbq_parameters}};
    } else {
        nlohmann::json hnswsq_parameters{{"base_quantization_type", base_quantization_type},
                                            {"max_degree", max_degree}, 
                                            {"ef_construction", ef_construction},
                                            {"build_thread_count", 1},
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
               void* invalid, bool reverse_filter) {
    return knn_search(index_handler, query_vector, dim, topk, dist, ids, result_size,
                      ef_search, need_extra_info, extra_infos, invalid, reverse_filter);
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

} //namespace obvectorlib