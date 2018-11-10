#include <algorithm>
#include <vector>

#include "caffe/layers/relu3_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLU3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype threshold_ = this->layer_param_.relu3_param().threshold();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu3_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], threshold_)
        + negative_slope * std::min(bottom_data[i], threshold_);
  }
}

template <typename Dtype>
void ReLU3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Dtype threshold_ = this->layer_param_.relu3_param().threshold();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu3_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > threshold_)
          + negative_slope * (bottom_data[i] <= threshold_));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLU3Layer);
#endif

INSTANTIATE_CLASS(ReLU3Layer);
REGISTER_LAYER_CLASS(ReLU3);

}  // namespace caffe
