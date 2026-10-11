#include <metal_stdlib>
using namespace metal;

[[visible]] uint visible_function_add_one(uint value) {
    return value + 1u;
}
