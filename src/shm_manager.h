// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <boost/interprocess/managed_external_buffer.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <iostream>
#include <vector>

#pragma once

namespace bi = boost::interprocess;


namespace triton { namespace backend { namespace python {
class SharedMemory {
 public:
  template <typename T>
  struct AllocatedSharedMemory {
    T* ptr;
    bi::managed_external_buffer::handle_t handle;
  };

  SharedMemory(
      const std::string& shm_region_name, int64_t shm_default_size, bool create,
      int64_t shm_growth_size);

  template <typename T>
  std::unique_ptr<AllocatedSharedMemory<T>> Allocate(size_t number)
  {
  }

  template <typename T>
  std::unique_ptr<AllocatedSharedMemory<T>> Allocate()
  {
    bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};

    GrowIfNeeded(sizeof(T));
    void* memory = managed_buffer_->allocate(sizeof(T));

    auto allocated_shared_memory = std::make_unique<AllocatedSharedMemory<T>>();
    allocated_shared_memory->ptr = reinterpret_cast<T*>(memory);
    allocated_shared_memory->handle =
        managed_buffer_->get_handle_from_address(memory);
    return allocated_shared_memory;
  }

  template <typename T>
  void Deallocate(std::unique_ptr<AllocatedSharedMemory<T>>&& allocated_memory)
  {
    bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};

    GrowIfNeeded(0);
    size_t shm_map_sizes = old_shm_maps_.size();
    void* ptr =
        managed_buffer_->get_address_from_handle(allocated_memory->handle);
    managed_buffer_->deallocate(ptr);
  }

  template <typename T>
  std::unique_ptr<AllocatedSharedMemory<T>> Load(
      bi::managed_external_buffer::handle_t handle)
  {
    bi::scoped_lock<bi::interprocess_mutex> gaurd{*shm_mutex_};

    GrowIfNeeded(0);
    void* address = managed_buffer_->get_address_from_handle(handle);
    std::unique_ptr<AllocatedSharedMemory<T>> allocated_shared_memory =
        std::make_unique<AllocatedSharedMemory<T>>();
    allocated_shared_memory->ptr = reinterpret_cast<T>(address);
    allocated_shared_memory->handle = handle;

    return allocated_shared_memory;
  }

  size_t FreeMemory();
  ~SharedMemory() noexcept(false);

 private:
  std::string shm_region_name_;
  std::unique_ptr<bi::managed_external_buffer> managed_buffer_;
  std::unique_ptr<bi::shared_memory_object> shm_obj_;
  std::shared_ptr<bi::mapped_region> shm_map_;
  std::vector<std::shared_ptr<bi::mapped_region>> old_shm_maps_;
  int64_t default_size_;
  int64_t growth_size_;
  size_t current_capacity_;
  bi::interprocess_mutex* shm_mutex_;
  size_t shm_growth_bytes_;
  bool create_;
  void GrowIfNeeded(size_t bytes);
};

}}}  // namespace triton::backend::python
