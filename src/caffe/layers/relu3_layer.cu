#include <algorithm>
#include <vector>

#include "caffe/layers/relu3_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLU3Forward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype threshold_) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold_ ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ReLU3Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu3_param().negative_slope();
  Dtype threshold_ = this->layer_param_.relu2_param().threshold();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLU3Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, threshold_);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLU3Backward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope,Dtype threshold_) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > threshold_)
        + (in_data[index] <= threshold_) * negative_slope);
  }
}

template <typename Dtype>
void ReLU3Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu3_param().negative_slope();
    Dtype threshold_ = this->layer_param_.relu2_param().threshold();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLU3Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope, threshold_);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLU3Layer);


}  // namespace caffe
