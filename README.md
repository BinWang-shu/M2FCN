# Multi-stage Multi-recursive-input Fully Convolutional Networks for Neuronal Boundary Detection
This is an official implementation for [Multi-stage Multi-recursive-input Fully Convolutional Networks for Neuronal Boundary Detection](http://xueshu.baidu.com/s?wd=paperuri%3A%28c23476cd2c605dfdb62cc4c1bbb1094a%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1703.08493&ie=utf-8&sc_us=4634636395865895406) based on caffe.

## Demo
  1. You need compile the caffe.
   ```Bash
   make
   make pycaffe
   ```  
  2. Changing the path to your datasets in ./examples/final_results/train_val.prototxt
  3. To trian the model,run
   ```python
   cd ./examples/final_results
   python solve.py
   ```
  4. To test the model,run
   ```python
   python result.py
   ```
