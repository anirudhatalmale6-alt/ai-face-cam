/*
 * GridSample3D custom operator for ONNX Runtime 1.24.x
 * Implements trilinear interpolation for 5D tensors
 * align_corners=False, padding_mode=zeros, mode=bilinear (trilinear for 3D)
 */
#ifdef _WIN32
#define ORT_API_MANUAL_INIT
#include "onnxruntime_c_api.h"
#define EXPORT __declspec(dllexport)
#else
#include "onnxruntime_c_api.h"
#define EXPORT __attribute__((visibility("default")))
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>

static const OrtApi* g_ort = NULL;

/* CreateKernel */
static void* ORT_API_CALL MyCreateKernel(const struct OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    int* kernel = (int*)malloc(sizeof(int));
    *kernel = 1;
    return kernel;
}

/* GetName */
static const char* ORT_API_CALL MyGetName(const struct OrtCustomOp* op) {
    return "GridSample3D";
}

/* GetExecutionProviderType */
static const char* ORT_API_CALL MyGetExecutionProviderType(const struct OrtCustomOp* op) {
    return NULL;
}

/* GetInputType */
static ONNXTensorElementDataType ORT_API_CALL MyGetInputType(const struct OrtCustomOp* op, size_t index) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

/* GetInputTypeCount */
static size_t ORT_API_CALL MyGetInputTypeCount(const struct OrtCustomOp* op) {
    return 2;
}

/* GetOutputType */
static ONNXTensorElementDataType ORT_API_CALL MyGetOutputType(const struct OrtCustomOp* op, size_t index) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

/* GetOutputTypeCount */
static size_t ORT_API_CALL MyGetOutputTypeCount(const struct OrtCustomOp* op) {
    return 1;
}

/* KernelCompute - trilinear interpolation with align_corners=False, padding_mode=zeros */
static void ORT_API_CALL MyKernelCompute(void* op_kernel, OrtKernelContext* context) {
    const OrtValue* input_val = NULL;
    const OrtValue* grid_val = NULL;
    g_ort->KernelContext_GetInput(context, 0, &input_val);
    g_ort->KernelContext_GetInput(context, 1, &grid_val);

    OrtTensorTypeAndShapeInfo* input_info = NULL;
    OrtTensorTypeAndShapeInfo* grid_info = NULL;
    g_ort->GetTensorTypeAndShape(input_val, &input_info);
    g_ort->GetTensorTypeAndShape(grid_val, &grid_info);

    int64_t input_shape[5], grid_shape[5];
    g_ort->GetDimensions(input_info, input_shape, 5);
    g_ort->GetDimensions(grid_info, grid_shape, 5);
    g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
    g_ort->ReleaseTensorTypeAndShapeInfo(grid_info);

    int64_t N = input_shape[0], C = input_shape[1], D = input_shape[2], H = input_shape[3], W = input_shape[4];
    int64_t D2 = grid_shape[1], H2 = grid_shape[2], W2 = grid_shape[3];

    int64_t out_shape[5] = {N, C, D2, H2, W2};
    OrtValue* output_val = NULL;
    g_ort->KernelContext_GetOutput(context, 0, out_shape, 5, &output_val);

    const float* input_data = NULL;
    const float* grid_data = NULL;
    float* output_data = NULL;
    g_ort->GetTensorData(input_val, (const void**)&input_data);
    g_ort->GetTensorData(grid_val, (const void**)&grid_data);
    g_ort->GetTensorMutableData(output_val, (void**)&output_data);

    for (int64_t n = 0; n < N; n++) {
        for (int64_t d2 = 0; d2 < D2; d2++) {
            for (int64_t h2 = 0; h2 < H2; h2++) {
                for (int64_t w2 = 0; w2 < W2; w2++) {
                    int64_t grid_idx = ((n * D2 + d2) * H2 + h2) * W2 + w2;
                    float gx = grid_data[grid_idx * 3 + 0];
                    float gy = grid_data[grid_idx * 3 + 1];
                    float gz = grid_data[grid_idx * 3 + 2];

                    /* align_corners=False: map [-1,1] to [-0.5, size-0.5] */
                    float fx = ((gx + 1.0f) * (float)W - 1.0f) * 0.5f;
                    float fy = ((gy + 1.0f) * (float)H - 1.0f) * 0.5f;
                    float fz = ((gz + 1.0f) * (float)D - 1.0f) * 0.5f;

                    int x0 = (int)floorf(fx), y0 = (int)floorf(fy), z0 = (int)floorf(fz);
                    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
                    float wx = fx - (float)x0, wy = fy - (float)y0, wz = fz - (float)z0;

                    for (int64_t c = 0; c < C; c++) {
                        const float* vol = input_data + (n * C + c) * D * H * W;

                        /* padding_mode=zeros: return 0 for out-of-bounds */
#define SAFE(z,y,x) (((z)>=0 && (z)<D && (y)>=0 && (y)<H && (x)>=0 && (x)<W) ? vol[(z)*H*W+(y)*W+(x)] : 0.0f)
                        float c000 = SAFE(z0, y0, x0);
                        float c001 = SAFE(z0, y0, x1);
                        float c010 = SAFE(z0, y1, x0);
                        float c011 = SAFE(z0, y1, x1);
                        float c100 = SAFE(z1, y0, x0);
                        float c101 = SAFE(z1, y0, x1);
                        float c110 = SAFE(z1, y1, x0);
                        float c111 = SAFE(z1, y1, x1);
#undef SAFE

                        float c00 = c000*(1-wx) + c001*wx;
                        float c01 = c010*(1-wx) + c011*wx;
                        float c10 = c100*(1-wx) + c101*wx;
                        float c11 = c110*(1-wx) + c111*wx;
                        float c_0 = c00*(1-wy) + c01*wy;
                        float c_1 = c10*(1-wy) + c11*wy;
                        float val = c_0*(1-wz) + c_1*wz;

                        output_data[((n*C+c)*D2+d2)*H2*W2 + h2*W2 + w2] = val;
                    }
                }
            }
        }
    }
}

/* KernelDestroy */
static void ORT_API_CALL MyKernelDestroy(void* op_kernel) {
    free(op_kernel);
}

/* GetInputCharacteristic */
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL MyGetInputCharacteristic(const struct OrtCustomOp* op, size_t index) {
    return INPUT_OUTPUT_REQUIRED;
}

/* GetOutputCharacteristic */
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL MyGetOutputCharacteristic(const struct OrtCustomOp* op, size_t index) {
    return INPUT_OUTPUT_REQUIRED;
}

/* GetInputMemoryType */
static OrtMemType ORT_API_CALL MyGetInputMemoryType(const struct OrtCustomOp* op, size_t index) {
    return OrtMemTypeDefault;
}

static int ORT_API_CALL MyGetVariadicInputMinArity(const struct OrtCustomOp* op) { return 1; }
static int ORT_API_CALL MyGetVariadicInputHomogeneity(const struct OrtCustomOp* op) { return 1; }
static int ORT_API_CALL MyGetVariadicOutputMinArity(const struct OrtCustomOp* op) { return 1; }
static int ORT_API_CALL MyGetVariadicOutputHomogeneity(const struct OrtCustomOp* op) { return 1; }
static int ORT_API_CALL MyGetStartVersion(const struct OrtCustomOp* op) { return 1; }
static int ORT_API_CALL MyGetEndVersion(const struct OrtCustomOp* op) { return 20; }

/*
 * Struct layout for ORT 1.26.0 (API version 21):
 * version, CreateKernel, GetName, GetExecutionProviderType,
 * GetInputType, GetInputTypeCount, GetOutputType, GetOutputTypeCount,
 * KernelCompute, KernelDestroy,
 * GetInputCharacteristic, GetOutputCharacteristic,
 * GetInputMemoryType,
 * GetVariadicInputMinArity, GetVariadicInputHomogeneity,
 * GetVariadicOutputMinArity, GetVariadicOutputHomogeneity,
 * CreateKernelV2, KernelComputeV2, InferOutputShapeFn,
 * GetStartVersion, GetEndVersion,
 * GetMayInplace, ReleaseMayInplace, GetAliasMap, ReleaseAliasMap
 */
static struct OrtCustomOp grid_sample_3d_op = {
    ORT_API_VERSION,
    MyCreateKernel,
    MyGetName,
    MyGetExecutionProviderType,
    MyGetInputType,
    MyGetInputTypeCount,
    MyGetOutputType,
    MyGetOutputTypeCount,
    MyKernelCompute,
    MyKernelDestroy,
    MyGetInputCharacteristic,
    MyGetOutputCharacteristic,
    MyGetInputMemoryType,
    MyGetVariadicInputMinArity,
    MyGetVariadicInputHomogeneity,
    MyGetVariadicOutputMinArity,
    MyGetVariadicOutputHomogeneity,
    NULL,  /* CreateKernelV2 */
    NULL,  /* KernelComputeV2 */
    NULL,  /* InferOutputShapeFn */
    MyGetStartVersion,
    MyGetEndVersion,
    NULL,  /* GetMayInplace */
    NULL,  /* ReleaseMayInplace */
    NULL,  /* GetAliasMap */
    NULL,  /* ReleaseAliasMap */
};

EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    g_ort = api_base->GetApi(ORT_API_VERSION);
    OrtCustomOpDomain* domain = NULL;
    OrtStatus* status = g_ort->CreateCustomOpDomain("", &domain);
    if (status) return status;
    status = g_ort->CustomOpDomain_Add(domain, &grid_sample_3d_op);
    if (status) return status;
    status = g_ort->AddCustomOpDomain(options, domain);
    return status;
}
