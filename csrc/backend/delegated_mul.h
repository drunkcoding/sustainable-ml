#pragma once

#include <torch/extension.h>
#include <thread>
#include <unordered_map>
#include <iostream>
#include <cuda_runtime_api.h>
#include <chrono> 

template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
struct DoNothingDeleter {
    void operator()(T* ptr) const {}
};

class MatMulThreadPool {
 public:
  MatMulThreadPool(int num_threads) : num_threads_(num_threads) {
    num_gpus_ = torch::cuda::device_count();
    gpu_mutex_.reserve(num_gpus_);
    gpu_condition_.reserve(num_gpus_);
    // gpu_tensors_.reserve(num_gpus_);

    gpu_task_count_.reserve(num_gpus_);
    gpu_buffer_count_.reserve(num_gpus_);
    gpu_output_count_.reserve(num_gpus_);

    std::cout << "Number of GPUs: " << num_gpus_ << std::endl;
    for (int i = 0; i < num_gpus_; i++) {
      gpu_threads_[i] = std::vector<std::thread>();
      gpu_tasks_[i] = std::queue<std::function<void()>>();
      gpu_task_count_[i] = 0;
      gpu_buffer_count_[i] = 0;
      gpu_output_count_[i] = 0;
      gpu_tensors_[i] = torch::zeros({1}, torch::kCPU);
      gpu_buffers_[i] = std::vector<torch::Tensor>(1000);
      gpu_outputs_[i] = std::vector<torch::Tensor>(1000);
      for (int j = 0; j < 1000; j++) {
        gpu_buffers_[i][j] = torch::zeros({1}, torch::kCPU);
        gpu_outputs_[i][j] = torch::zeros({1}, torch::kCPU);
      }

      std::cout << "GPU " << i << " has " << num_threads_ << " threads" << std::endl;

      for (int j = 0; j < num_threads_; j++) {
        gpu_threads_[i].emplace_back([this, i] {
          while (true) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(this->gpu_mutex_[i]);
              this->gpu_condition_[i].wait(lock, [this, i] { return this->stop_ || !this->gpu_tasks_[i].empty(); });
              if (this->stop_ && this->gpu_tasks_[i].empty()) {
                return;
              }
              // std::cout << "GPU " << i << " thread " << std::this_thread::get_id() << " is running" << std::endl;
              task = std::move(this->gpu_tasks_[i].front());
              this->gpu_tasks_[i].pop();
              // std::cout << "GPU " << i << " thread " << std::this_thread::get_id() << " got a task" << std::endl;
            }
            task();
            {
              std::unique_lock<std::mutex> lock(this->gpu_mutex_[i]);
              gpu_task_count_[i]--;
              // std::cout << "GPU " << i << " thread " << std::this_thread::get_id() << " is done" << std::endl;
            }
            this->gpu_condition_[i].notify_all();
          }
        });
      }
    }
    std::cout << "MatMulThreadPool initialized" << std::endl;
  }

  ~MatMulThreadPool() {
    {
      // std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    
    for (int i = 0; i < num_gpus_; i++) {
      for (int j = 0; j < num_threads_; j++) {
        gpu_condition_[i].notify_all();
        gpu_threads_[i][j].join();
      }
    }
  }

  // void Enqueue(std::function<void()> task, int gid) {
  //   {
  //     std::unique_lock<std::mutex> lock(gpu_mutex_[gid]);
  //     gpu_tasks_[gid].emplace(task);
  //   }
  //   gpu_condition_[gid].notify_one();
  // }

  // struct TensorMeta {
  //   std::vector<std::int64_t> shape;
  //   torch::TensorOptions options;
  //   void* data;
  // };

  void Enqueue(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c, int gid) {
    {
      auto gpu_device = torch::Device(torch::kCUDA, gid);
      // auto cpu_device = torch::Device(torch::kCPU);
      std::unique_lock<std::mutex> lock(gpu_mutex_[gid]);
      gpu_task_count_[gid]++;
      gpu_buffer_count_[gid]++;

      // TensorMeta a_meta{
      //   a.sizes().vec(),
      //   a.options(),
      //   a.data_ptr()
      // };

      // TensorMeta b_meta{
      //   b.sizes().vec(),
      //   b.options(),
      //   b.data_ptr()
      // };

      // auto func = std::bind(&MatMulThreadPool::MatMul, this, std::ref(a_meta), std::ref(b_meta), std::ref(c));
      // gpu_tasks_[gid].emplace(func);

      // std::cout << "GPU " << gid << a.sizes() << " " << b.sizes() << " " << c.sizes() << std::endl;

      cudaSetDevice(gid);

      if (!gpu_tensors_[gid].is_cuda()) {
        void* a_ptr = nullptr;
        auto a_byte_size = a.numel() * a.element_size();
        cudaMalloc(&a_ptr, a_byte_size);
        cudaMemcpy(a_ptr, a.data_ptr(), a_byte_size, cudaMemcpyHostToDevice);
        gpu_tensors_[gid].set_data(torch::from_blob(a_ptr, a.sizes(), CudaDeleter<void>(), a.options().device(gpu_device)));
      }
      torch::Tensor& a_tensor = gpu_tensors_[gid];
      
      std::cout << "GPU " << gid << a_tensor.sizes() << " " << b.sizes() << " " << c.sizes() << std::endl;

      void* b_ptr = nullptr;
      auto b_byte_size = b.numel() * b.element_size();
      cudaMalloc(&b_ptr, b_byte_size);
      cudaMemcpy(b_ptr, b.data_ptr(), b_byte_size, cudaMemcpyHostToDevice);
      // auto tensor = torch::from_blob(b_ptr, b.sizes(), CudaDeleter<void>(), b.options().device(gpu_device));
      gpu_buffers_[gid][gpu_buffer_count_[gid] - 1].set_data(torch::from_blob(b_ptr, b.sizes(), CudaDeleter<void>(), b.options().device(gpu_device)));
      torch::Tensor& b_tensor = gpu_buffers_[gid][gpu_buffer_count_[gid] - 1];

      // std::cout << "GPU " << gid << a_tensor.sizes() << " " << b_tensor.sizes() << " " << c.sizes() << std::endl;

      auto func = std::bind(&MatMulThreadPool::MatMul, this, std::ref(a_tensor), std::ref(b_tensor), std::ref(c));
      gpu_tasks_[gid].emplace(func);

      // std::shared_ptr<torch::Tensor> a_tensor_ptr = std::make_shared<torch::Tensor>(
      //   torch::from_blob(a_ptr, a.sizes(), CudaDeleter<void>(), a.options())
      // );
      // std::shared_ptr<torch::Tensor> b_tensor_ptr = std::make_shared<torch::Tensor>(
      //   torch::from_blob(b_ptr, b.sizes(), CudaDeleter<void>(), b.options())
      // );

      // std::cout << "GPU " << gid << " thread " << std::this_thread::get_id() << " is running" << std::endl;

      // gpu_tasks_[gid].emplace([this, gid, a_tensor_ptr, b_tensor_ptr, &c] {
      //   std::cout << "GPU " << gid << a_tensor_ptr->sizes() << " " << b_tensor_ptr->sizes() << " " << c.sizes() << std::endl;
      //   // c.set_data(a.to(gpu_device).matmul(b.to(gpu_device)).to(cpu_device));
      //   auto out = torch::matmul(*a_tensor_ptr, *b_tensor_ptr);
      //   // c.set_data(torch::matmul(*a_tensor_ptr, *b_tensor_ptr).data());
      //   std::cout << "GPU " << gid << " thread " << std::this_thread::get_id() << " is done" << std::endl;
      // });


      // auto func = std::bind(&MatMulThreadPool::MatMul, this, std::ref(a), std::ref(b), std::ref(c));
      // gpu_tasks_[gid].emplace(func);
      // c.set_data(torch::matmul(a, b).data());
      // gpu_task_count_[gid]--;
    }
    gpu_condition_[gid].notify_all();
  }

  torch::Tensor WaitAll() {
    for (int i = 0; i < num_gpus_; i++) {
      std::unique_lock<std::mutex> lock(gpu_mutex_[i]);
      gpu_condition_[i].wait(lock, [this, i] { return this->gpu_task_count_[i] == 0; });
    }

    std::vector<torch::Tensor> outputs;
    std::vector<int> pivots(num_gpus_, 0);

    for (int k = 0; k < 1000; k++) {
      for (int i = 0; i < num_gpus_; i++) {
        if (pivots[i] < gpu_output_count_[i]) {
          outputs.push_back(gpu_outputs_[i][pivots[i]]);
          pivots[i]++;
        }
      }
    }
    
    for (int i = 0; i < num_gpus_; i++) {
      gpu_output_count_[i] = 0;
      gpu_buffer_count_[i] = 0;
      gpu_tensors_[i] = torch::zeros({1}, torch::kCPU);
    }

    return torch::cat(outputs, -1);
  }

  // void WaitGPU(int gid) {
  //   std::unique_lock<std::mutex> lock(gpu_mutex_[gid]);
  //   gpu_condition_[gid].wait(lock, [this, gid] { return this->gpu_task_count_[gid] == 0; });
  // }

private:

  void Concatenate(std::vector<torch::Tensor> &outputs) {
    // use torch_index_copy_ to concatenate tensors
    
  }

  // void MatMul(TensorMeta& a_meta, TensorMeta& b_meta, torch::Tensor &c) {
  void MatMul(torch::Tensor& a, torch::Tensor& b, torch::Tensor &c) {

    // std::cout << "GPU xxxx " << a.sizes() << " " << b.sizes() << std::endl;
    int gid = a.device().index();

    auto start_time = std::chrono::high_resolution_clock::now();
    auto out = torch::matmul(a, b);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "GPU " << gid << " thread " << std::this_thread::get_id() << " is done in " << duration.count() << " microseconds" << std::endl;

    // auto a = torch::from_blob(a_meta.data, a_meta.shape, CudaDeleter<void>(), a_meta.options);
    // auto b = torch::from_blob(b_meta.data, b_meta.shape, CudaDeleter<void>(), b_meta.options);

    // std::cout << "GPU ..." << std::endl;

    gpu_outputs_[gid][gpu_output_count_[gid]++].set_data(out.to("cpu"));
  }

  // void MatMul(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c) {
  //   std::cout << "MatMul" << std::endl;
  //   auto out = torch::matmul(a, b);
  //   // std::cout << "GPU " << gpu_device.index() << " thread " << std::this_thread::get_id() << " is done" << std::endl;
  // }

 private:
  std::unordered_map<int, std::vector<std::thread>> gpu_threads_;
  std::unordered_map<int, std::queue<std::function<void()>>> gpu_tasks_;
  std::vector<int> gpu_task_count_;
  std::unordered_map<int, std::mutex> gpu_mutex_;
  std::unordered_map<int, std::condition_variable> gpu_condition_;
  bool stop_ = false;
  int num_threads_; // number of threads for each GPU
  int num_gpus_;

  std::unordered_map<int, torch::Tensor> gpu_tensors_;
  std::unordered_map<int, std::vector<torch::Tensor>> gpu_buffers_;
  std::unordered_map<int, std::vector<torch::Tensor>> gpu_outputs_;
  std::vector<int> gpu_buffer_count_;
  std::vector<int> gpu_output_count_;
};