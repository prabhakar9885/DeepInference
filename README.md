
# Deep Inference
A C++ library to acceralarte the task of inference by liveraging the Nvidia GPU, without worrying about writing the GPU specific code.

# Prerequisites
- Nvidia GPU
- CUDA  v10.2
- cuDNN v7.6.5.32 for CUDA v10.2

# Contents
The repo has the following projects

| Project name | Description |
| ------------- | ------------- |
| DeepInferenceLib  | This library project contains the GPU specifc code.  |
| Dense network  | Contains a sample project with a conv net in it. |

# Todo
- [ ] Add support for *Pooling* layers.
- [ ] Add a Logger
- [ ] Document *How to load weights* ?
  - [ ] Before that, provide a clean, consistent and scalable way to load the weights without any dependency on NN architecture.
- [ ] Refactor the code for better readility and maintainability by following the *C++ best-practices and STLs*.
- [ ] Optimize the code for efficient use of Device and Host Memory
  - [ ] Check if we can use [Thrust, the CUDA C++ template library](https://docs.nvidia.com/cuda/thrust/index.html)
- [ ] Add *Unit-tests* and *Assert* statements whereever necessary.
- [ ] Brainstorm on the pros and cons(in terms of sacalability, maintainability and ease of use) of having a *ContextFactory*. If possible, get ride of the *ContextFactory*.
      
