#include "../ob_vsag_lib.h"
#include "default_allocator.h"
#include <random>
#include <dlfcn.h>
#include "../ob_vsag_lib_c.h"
#include <iostream>
#include "../default_logger.h"
#include "roaring/roaring64.h"
#include "vsag/iterator_context.h"
#include <stdio.h>
#include <stdlib.h>

class TestFilter : public obvectorlib::FilterInterface
{
public:
    TestFilter(roaring::api::roaring64_bitmap_t *bitmap) : bitmap_(bitmap) {}
    ~TestFilter() {}
    bool test(int64_t id) override { return roaring::api::roaring64_bitmap_contains(bitmap_, id); }
    bool test(const char* data) override { return true; }
public:
    roaring::api::roaring64_bitmap_t* bitmap_;
};

int64_t example() {
    std::cout<<"test hnsw_example: "<<std::endl;
    bool is_init = obvectorlib::is_init();
    //set_log_level(1);
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 1536;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";
    const char* const METRIC_IP = "ip";

    const char* const DATATYPE_FLOAT32 = "float32";
    void * test_ptr = default_allocator.Allocate(10);
    int ret_create_index = obvectorlib::create_index(index_handler,
                                                     obvectorlib::HNSW_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator);
   
    if (ret_create_index!=0) return 333;
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    int ret_build_index = obvectorlib::build_index(index_handler, vectors, ids, dim, num_vectors);

    int64_t num_size = 0;
    int ret_get_element = obvectorlib::get_index_number(index_handler, num_size);
    std::cout<<"after add index, size is "<<num_size<<" " <<ret_get_element<<std::endl;

    int inc_num = 10000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = i + num_vectors;
    }
 
    int ret_add_index = obvectorlib::add_index(index_handler, inc, ids2, dim,inc_num);
    ret_get_element = obvectorlib::get_index_number(index_handler, num_size);
    std::cout<<"after add index, size is "<<num_size<<" " <<ret_add_index<<std::endl;
    
    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();
    TestFilter testfilter(r1);
    const char *extra_info = nullptr;
    int ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, false/*need_extra_info*/, extra_info, &testfilter, false, false, nullptr, 1);
    
    roaring64_bitmap_add_range(r1, 0, 19800);

    ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, false/*need_extra_info*/, extra_info, &testfilter, false, false, nullptr, 0.01);
    const float *distances;
    // ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids, result_size, distances);
    for (int i = 0; i < result_size; i++) {
        std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
        // std::cout << "calres: " << result_ids[i] << " " << distances[i] << std::endl;
    }
    obvectorlib::delete_index(index_handler);
    free(test_ptr);
    return 0;
}

void
vsag::logger::ObDefaultLogger::SetLevel(Logger::Level log_level) {
    //
}

void
vsag::logger::ObDefaultLogger::Trace(const std::string& msg) {
    //
}

void
vsag::logger::ObDefaultLogger::Debug(const std::string& msg) {
    //
}

void
vsag::logger::ObDefaultLogger::Info(const std::string& msg) {
    //
}

void
vsag::logger::ObDefaultLogger::Warn(const std::string& msg) {
    //
}

void
vsag::logger::ObDefaultLogger::Error(const std::string& msg) {
    //
}

void
vsag::logger::ObDefaultLogger::Critical(const std::string& msg) {
    //
}

int example_so() {
    std::cout<<"test hnsw_example with dlopen: "<<std::endl;
    // Path to the dynamic library
    const char* lib_path = "./libob_vsag.so";  // Linux
    // const char* lib_path = "libexample.dylib";  // macOS

    // Open the dynamic library
    void* handle = dlopen(lib_path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        return EXIT_FAILURE;
    }
    

    obvectorlib::set_logger_ptr set_logger_c;
    LOAD_FUNCTION(handle, obvectorlib::set_logger_ptr, set_logger_c);
    //void* raw_memory = (void*)malloc(sizeof( vsag::logger::ObDefaultLogger));
    //vsag::logger::ObDefaultLogger* ob_logger = new (raw_memory)vsag::logger::ObDefaultLogger();
    //vsag::logger::ObDefaultLogger* ob_logger = new vsag::logger::ObDefaultLogger();
    //set_logger_c(ob_logger);

    //init
    //obvectorlib::is_init_ptr is_init_c;
    //LOAD_FUNCTION(handle, obvectorlib::is_init_ptr, is_init_c);
    //bool is_vsag_init_ = is_init_c();
    //std::cout << "is_vsag_init_: " << is_vsag_init_ << std::endl;

    //create index
    obvectorlib::create_index_ptr create_index_c;
    LOAD_FUNCTION(handle, obvectorlib::create_index_ptr, create_index_c);
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 128;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";
    const char* const DATATYPE_FLOAT32 = "float32";
    int ret_create_index = create_index_c(index_handler,
                                                     obvectorlib::HNSW_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator);

    //build index
    obvectorlib::build_index_ptr build_index_c;
    LOAD_FUNCTION(handle, obvectorlib::build_index_ptr, build_index_c);
    obvectorlib::get_index_number_ptr get_index_number_c;
    LOAD_FUNCTION(handle, obvectorlib::get_index_number_ptr, get_index_number_c);
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    int ret_build_index = build_index_c(index_handler, vectors, ids, dim, num_vectors);
    
    int64_t num_size = 0;
    int ret_get_element = get_index_number_c(index_handler, num_size);

    //add index
    obvectorlib::add_index_ptr add_index_c;
    LOAD_FUNCTION(handle, obvectorlib::add_index_ptr, add_index_c);
    int inc_num = 10000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = num_size+i;
    }
    
    int ret_add_index = add_index_c(index_handler, inc, ids2, dim,inc_num);
    ret_get_element = get_index_number_c(index_handler, num_size);
    
    //knn_search
    obvectorlib::knn_search_ptr knn_search_c;
    LOAD_FUNCTION(handle, obvectorlib::knn_search_ptr, knn_search_c);
    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();

    roaring::api::roaring64_bitmap_add(r1, 9999);
    roaring::api::roaring64_bitmap_add(r1, 1169);
    roaring::api::roaring64_bitmap_add(r1, 1285);

    int ret_knn_search = knn_search_c(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, r1, false);

    //serialize/deserialize
    obvectorlib::serialize_ptr serialize_c;
    LOAD_FUNCTION(handle, obvectorlib::serialize_ptr, serialize_c);
    obvectorlib::deserialize_bin_ptr deserialize_bin_c;
    LOAD_FUNCTION(handle, obvectorlib::deserialize_bin_ptr, deserialize_bin_c);
    const std::string dir = "./";
    int ret_serialize_single = serialize_c(index_handler,dir);
    int ret_deserilize_single_bin = deserialize_bin_c(index_handler,dir);
    obvectorlib::delete_index_c(index_handler);

    // Clean up
    dlclose(handle);
    
    return 0;
}

int64_t hnswsq_example() {
    std::cout<<"test hnswsq_example: "<<std::endl;
    bool is_init = obvectorlib::is_init();
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 128;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";
    const char* const METRIC_IP = "ip";

    const char* const DATATYPE_FLOAT32 = "float32";
    void * test_ptr = default_allocator.Allocate(10);
    int ret_create_index = obvectorlib::create_index(index_handler,
                                                     obvectorlib::HNSW_SQ_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator);
   
    if (ret_create_index!=0) return 333;
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i + num_vectors*10;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    int ret_build_index = obvectorlib::build_index(index_handler, vectors, ids, dim, num_vectors);

    int64_t num_size = 0;
    int ret_get_element = obvectorlib::get_index_number(index_handler, num_size);
    std::cout<<"after add index, size is "<<num_size<<" " <<ret_get_element<<std::endl;
    
    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();

    roaring::api::roaring64_bitmap_add(r1, 18);
    roaring::api::roaring64_bitmap_add(r1, 1169);
    roaring::api::roaring64_bitmap_add(r1, 1285);
    std::cout << "before search" << std::endl;
    TestFilter testfilter(r1);
    const char *extra_info = nullptr;
    int ret_knn_search = obvectorlib::knn_search(index_handler, query_vector, dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, false/*need_extra_info*/, extra_info, &testfilter);
    
    for (int i = 0; i < result_size; i++) {
        std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
    }
    int inc_num = 1000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = i + num_vectors*100;
    }

    // const std::string dir = "./";
    // int ret_serialize_single = obvectorlib::serialize(index_handler,dir);
    // int ret_deserilize_single_bin = 
    //                 obvectorlib::deserialize_bin(index_handler,dir);
    // ret_knn_search = obvectorlib::knn_search(index_handler, query_vector, dim, 10,
    //                                              result_dist,result_ids,result_size, 
    //                                              100, r1);
    // for (int i = 0; i < result_size; i++) {
    //     std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
    // }
    obvectorlib::delete_index(index_handler);
    free(test_ptr);
    return 0;
}

int64_t example_extra_info() {
    std::cout<<"test example_extra_info example: "<<std::endl;
    bool is_init = obvectorlib::is_init();
    //set_log_level(1);
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 1536;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";
    const char* const METRIC_IP = "ip";

    const char* const DATATYPE_FLOAT32 = "float32";
    void * test_ptr = default_allocator.Allocate(10);
    int extra_info_sz = 32;
    std::cout<<"test create_index: "<<std::endl;
    int ret_create_index = obvectorlib::create_index(index_handler,
                                                     obvectorlib::HGRAPH_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator,
                                                     extra_info_sz);

    if (ret_create_index!=0) return 333;
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    std::cout<<"test build_index: "<<std::endl;
    char extra_infos[extra_info_sz * num_vectors];
    for (int i = 0; i < extra_info_sz * num_vectors; i++) {
        extra_infos[i] = rand() % 9 + '0';
    }
    int ret_build_index = obvectorlib::build_index(index_handler, vectors, ids, dim, num_vectors, extra_infos);

    int64_t num_size = 0;
    std::cout<<"test get_index_number: "<<std::endl;
    int ret_get_element = obvectorlib::get_index_number(index_handler, num_size);
    std::cout<<"after add index, size is "<<num_size<<" " <<ret_get_element<<std::endl;

    int inc_num = 10000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = i + num_vectors;
    }

    std::cout<<"test add_index: "<<std::endl;
    char extra_info[extra_info_sz];
    for (int i = 0; i < extra_info_sz; i++) {
        extra_info[i] = rand() % 9 + '0';
    }
    int ret_add_index = obvectorlib::add_index(index_handler, inc, ids2, dim,inc_num, extra_info);
    ret_get_element = obvectorlib::get_index_number(index_handler, num_size);
    std::cout<<"after add index, size is "<<num_size<<" " <<ret_add_index<<std::endl;

    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();
    const char *extra_info_search = nullptr;
    TestFilter testfilter(r1);
    std::cout<<"test knn_search: "<<std::endl;
    int ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size,
                                                 100, true/*need_extra_info*/, extra_info_search, &testfilter);
    std::string s1(extra_info_search, extra_info_search + extra_info_sz);
    std::cout << s1 << std::endl;
    roaring64_bitmap_add_range(r1, 0, 19800);

    std::cout<<"test knn_search2: "<<std::endl;
    ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size,
                                                 100, true/*need_extra_info*/, extra_info_search, &testfilter);
    std::string s2(extra_info_search, extra_info_search + extra_info_sz);
    std::cout << s2 << std::endl;
    const float *distances = nullptr;
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids, result_size, distances);
    for (int i = 0; i < result_size; i++) {
        std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
        std::cout << "calres: " << result_ids[i] << " " << distances[i] << std::endl;
    }
    obvectorlib::delete_index(index_handler);
    free(test_ptr);
    return 0;
}

int64_t hgraph_iter_filter_example()
{
    std::cout<<"test hgraph_iter_filter_example: "<<std::endl;
    bool is_init = obvectorlib::is_init();
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 1536;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";
    const char* const METRIC_IP = "ip";

    const char* const DATATYPE_FLOAT32 = "float32";
    void * test_ptr = default_allocator.Allocate(10);
    int ret_create_index = obvectorlib::create_index(index_handler,
                                                     obvectorlib::HNSW_SQ_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator);
   
    if (ret_create_index!=0) return 333;
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    int inc_num = 10000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = i + num_vectors;
    }
    int ret_build_index = obvectorlib::build_index(index_handler, vectors, ids, dim, num_vectors);
 
    int ret_add_index = obvectorlib::add_index(index_handler, inc, ids2, dim,inc_num);
    
    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;
    
    // vsag::IteratorContext *filter_ctx = nullptr;
    void *iter_ctx = nullptr;

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();
    roaring64_bitmap_add_range(r1, 0, 500);
    TestFilter testfilter(r1);
    const float *distances;
    const char* extra_infos = nullptr;

    int ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, false, extra_infos, &testfilter, false, false, 0.97, iter_ctx, false);
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids, result_size, distances);
    std::cout << "-------- result1: --------" << std::endl;
    for (int i = 0; i < result_size; i++) {
        std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
        std::cout << "calres: " << result_ids[i] << " " << distances[i] << std::endl;
    }    
    

    const float *distances2;
    const float* result_dist2;
    const int64_t* result_ids2;
    int64_t result_size2 = 0;
    std::cout << "-------- result2: --------" << std::endl;
    ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist2,result_ids2,result_size2, 
                                                 100, false, extra_infos, &testfilter, false, false, 0.97, iter_ctx, false);
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids2, result_size2, distances2);
    for (int i = 0; i < result_size2; i++) {
        std::cout << "result: " << result_ids2[i] << " " << result_dist2[i] << std::endl;
        std::cout << "calres: " << result_ids2[i] << " " << distances2[i] << std::endl;
    }

    const float *distances3;
    const float* result_dist3;
    const int64_t* result_ids3;
    int64_t result_size3 = 0;
    std::cout << "-------- result3: --------" << std::endl;
    ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 20,
                                                 result_dist3,result_ids3,result_size3, 
                                                 100, false, extra_infos, &testfilter, false, false, 0.97, iter_ctx, true);
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids3, result_size3, distances3);
    for (int i = 0; i < result_size3; i++) {
        std::cout << "result: " << result_ids3[i] << " " << result_dist3[i] << std::endl;
        std::cout << "calres: " << result_ids3[i] << " " << distances3[i] << std::endl;
    }
    obvectorlib::delete_iter_ctx(iter_ctx);
    obvectorlib::delete_index(index_handler);
    free(test_ptr);
    return 0;
}

int64_t hnsw_iter_filter_example()
{
    std::cout<<"test iter_filter_example: "<<std::endl;
    bool is_init = obvectorlib::is_init();
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 1536;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";
    const char* const METRIC_IP = "ip";

    const char* const DATATYPE_FLOAT32 = "float32";
    void * test_ptr = default_allocator.Allocate(10);
    int ret_create_index = obvectorlib::create_index(index_handler,
                                                     obvectorlib::HNSW_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator);
   
    if (ret_create_index!=0) return 333;
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    int inc_num = 10000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = i + num_vectors;
    }
 
    int ret_add_index = obvectorlib::add_index(index_handler, inc, ids2, dim,inc_num);
    
    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;
    
    // vsag::IteratorContext *filter_ctx = nullptr;
    void *iter_ctx = nullptr;

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();
    roaring64_bitmap_add_range(r1, 0, 500);
    TestFilter testfilter(r1);
    const float *distances;
    const char* extra_infos = nullptr;

    int ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, false, extra_infos, &testfilter, false, false, 0.97, iter_ctx, false);
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids, result_size, distances);
    std::cout << "-------- result1: --------" << std::endl;
    for (int i = 0; i < result_size; i++) {
        std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
        std::cout << "calres: " << result_ids[i] << " " << distances[i] << std::endl;
    }    
    

    const float *distances2;
    const float* result_dist2;
    const int64_t* result_ids2;
    int64_t result_size2 = 0;
    std::cout << "-------- result2: --------" << std::endl;
    ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 10,
                                                 result_dist2,result_ids2,result_size2, 
                                                 100, false, extra_infos, &testfilter, false, false, 0.97, iter_ctx, false);
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids2, result_size2, distances2);
    for (int i = 0; i < result_size2; i++) {
        std::cout << "result: " << result_ids2[i] << " " << result_dist2[i] << std::endl;
        std::cout << "calres: " << result_ids2[i] << " " << distances2[i] << std::endl;
    }

    const float *distances3;
    const float* result_dist3;
    const int64_t* result_ids3;
    int64_t result_size3 = 0;
    std::cout << "-------- result3: --------" << std::endl;
    ret_knn_search = obvectorlib::knn_search(index_handler, vectors+dim*(num_vectors-1), dim, 20,
                                                 result_dist3,result_ids3,result_size3, 
                                                 100, false, extra_infos, &testfilter, false, false, 0.97, iter_ctx, true);
    ret_knn_search = obvectorlib::cal_distance_by_id(index_handler, vectors+dim*(num_vectors-1), result_ids3, result_size3, distances3);
    for (int i = 0; i < result_size3; i++) {
        std::cout << "result: " << result_ids3[i] << " " << result_dist3[i] << std::endl;
        std::cout << "calres: " << result_ids3[i] << " " << distances3[i] << std::endl;
    }
    obvectorlib::delete_iter_ctx(iter_ctx);
    obvectorlib::delete_index(index_handler);
    free(test_ptr);
    return 0;
}

int64_t hnswbq_example() {
    std::cout<<"test hns_bq_example: "<<std::endl;
    bool is_init = obvectorlib::is_init();
    obvectorlib::VectorIndexPtr index_handler = NULL;
    int dim = 128;
    int max_degree = 16;
    int ef_search = 200;
    int ef_construction = 100;
    DefaultAllocator default_allocator;
    const char* const METRIC_L2 = "l2";

    const char* const DATATYPE_FLOAT32 = "float32";
    void * test_ptr = default_allocator.Allocate(10);
    int ret_create_index = obvectorlib::create_index(index_handler,
                                                     obvectorlib::HNSW_BQ_TYPE,
                                                     DATATYPE_FLOAT32,
                                                     METRIC_L2,
                                                     dim,
                                                     max_degree,
                                                     ef_construction,
                                                     ef_search,
                                                     &default_allocator);
   
    if (ret_create_index!=0) return 333;
    int num_vectors = 10000;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i + num_vectors*10;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    int ret_build_index = obvectorlib::build_index(index_handler, vectors, ids, dim, num_vectors);

    int64_t num_size = 0;
    int ret_get_element = obvectorlib::get_index_number(index_handler, num_size);
    std::cout<<"after add index, size is "<<num_size<<" " <<ret_get_element<<std::endl;
    
    const float* result_dist;
    const int64_t* result_ids;
    int64_t result_size = 0;
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    roaring::api::roaring64_bitmap_t* r1 = roaring::api::roaring64_bitmap_create();

    roaring::api::roaring64_bitmap_add(r1, 18);
    roaring::api::roaring64_bitmap_add(r1, 1169);
    roaring::api::roaring64_bitmap_add(r1, 1285);
    std::cout << "before search" << std::endl;
    TestFilter testfilter(r1);
    const char *extra_info = nullptr;
    int ret_knn_search = obvectorlib::knn_search(index_handler, query_vector, dim, 10,
                                                 result_dist,result_ids,result_size, 
                                                 100, false/*need_extra_info*/, extra_info, &testfilter);
    
    for (int i = 0; i < result_size; i++) {
        std::cout << "result: " << result_ids[i] << " " << result_dist[i] << std::endl;
    }
    int inc_num = 1000;
    auto inc = new float[dim * inc_num];
    for (int64_t i = 0; i < dim * inc_num; ++i) {
        inc[i] = distrib_real(rng);
    }
    auto ids2 = new int64_t[inc_num];
    for (int64_t i = 0; i < inc_num; ++i) {
        ids2[i] = i + num_vectors*100;
    }
    obvectorlib::delete_index(index_handler);
    free(test_ptr);
    return 0;
}

int
main() {
    hnswsq_example();
    example();
    //example_extra_info();
    hnsw_iter_filter_example();
    hgraph_iter_filter_example();
    hnswbq_example();
    return 0;
}
