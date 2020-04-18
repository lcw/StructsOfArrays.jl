import CuArrays
import CuArrays: CuArray
_type_with_eltype(::Type{<:CuArray}, T, N) = CuArray{T, N, Nothing}
_type(::Type{<:CuArray}) = CuArray
_device(::Type{<:CuArray}) = CUDA()

import CUDAnative
import CUDAnative: CuDeviceArray
_type_with_eltype(::Type{<:CuDeviceArray}, T, N) = CuDeviceArray{T, N}
_type(::Type{<:CuDeviceArray}) = CuDeviceArray
