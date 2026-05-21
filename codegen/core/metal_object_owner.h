#pragma once

namespace codegen {

template <typename T>
class MetalOwnedObject {
public:
    MetalOwnedObject() = default;
    explicit MetalOwnedObject(T* object) noexcept : object_(object) {}
    ~MetalOwnedObject() { reset(); }

    MetalOwnedObject(const MetalOwnedObject&) = delete;
    MetalOwnedObject& operator=(const MetalOwnedObject&) = delete;

    MetalOwnedObject(MetalOwnedObject&& other) noexcept
        : object_(other.release()) {}

    MetalOwnedObject& operator=(MetalOwnedObject&& other) noexcept {
        if (this != &other) reset(other.release());
        return *this;
    }

    T* get() const noexcept { return object_; }
    T* operator->() const noexcept { return object_; }
    explicit operator bool() const noexcept { return object_ != nullptr; }

    T* release() noexcept {
        T* object = object_;
        object_ = nullptr;
        return object;
    }

    void reset(T* object = nullptr) noexcept {
        if (object_) object_->release();
        object_ = object;
    }

private:
    T* object_ = nullptr;
};

} // namespace codegen
