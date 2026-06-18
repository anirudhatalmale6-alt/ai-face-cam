/*
 * GridSample3D custom operator for ONNX Runtime
 * Implements trilinear interpolation for 5D tensors
 * No TensorRT dependency - pure C implementation
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

typedef struct {
    int dummy;
} GridSample3DKernel;

static void* CreateKernel(const OrtApi* api, const OrtKernelInfo* info) {
    GridSample3DKernel* kernel = (GridSample3DKernel*)malloc(sizeof(GridSample3DKernel));
    kernel->dummy = 0;
    return kernel;
}

static const char* GetName(const OrtCustomOp* op) {
    return "GridSample3D";
}

static ONNXTensorElementDataType GetInputType(const OrtCustomOp* op, size_t index) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t GetInputTypeCount(const OrtCustomOp* op) {
    return 2;
}

static ONNXTensorElementDataType GetOutputType(const OrtCustomOp* op, size_t index) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t GetOutputTypeCount(const OrtCustomOp* op) {
    return 1;
}

static OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(const OrtCustomOp* op, size_t index) {
    return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(const OrtCustomOp* op, size_t index) {
    return INPUT_OUTPUT_REQUIRED;
}

static int GetVariadicInputMinArity(const OrtCustomOp* op) { return 1; }
static int GetVariadicOutputMinArity(const OrtCustomOp* op) { return 1; }
static int GetVariadicInputHomogeneity(const OrtCustomOp* op) { return 1; }
static int GetVariadicOutputHomogeneity(const OrtCustomOp* op) { return 1; }

static inline float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static OrtStatusPtr Compute(void* kernel_ptr, OrtKernelContext* context) {
    const OrtValue* input_val = NULL;
    const OrtValue* grid_val = NULL;
    g_ort->KernelContext_GetInput(context, 0, &input_val);
    g_ort->KernelContext_GetInput(context, 1, &grid_val);

    OrtTensorTypeAndShapeInfo* input_info = NULL;
    OrtTensorTypeAndShapeInfo* grid_info = NULL;
    g_ort->GetTensorTypeAndShape(input_val, &input_info);
    g_ort->GetTensorTypeAndShape(grid_val, &grid_info);

    size_t input_ndim = 0, grid_ndim = 0;
    g_ort->GetDimensionsCount(input_info, &input_ndim);
    g_ort->GetDimensionsCount(grid_info, &grid_ndim);

    int64_t input_shape[5], grid_shape[5];
    g_ort->GetDimensions(input_info, input_shape, input_ndim);
    g_ort->GetDimensions(grid_info, grid_shape, grid_ndim);

    g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
    g_ort->ReleaseTensorTypeAndShapeInfo(grid_info);

    /* input: (N, C, D, H, W) */
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    /* grid: (N, D2, H2, W2, 3) */
    int64_t D2 = grid_shape[1];
    int64_t H2 = grid_shape[2];
    int64_t W2 = grid_shape[3];

    int64_t out_shape[5] = {N, C, D2, H2, W2};
    OrtValue* output_val = NULL;
    g_ort->KernelContext_GetOutput(context, 0, out_shape, 5, &output_val);

    const float* input_data = NULL;
    const float* grid_data = NULL;
    float* output_data = NULL;
    g_ort->GetTensorData(input_val, (const void**)&input_data);
    g_ort->GetTensorData(grid_val, (const void**)&grid_data);
    g_ort->GetTensorMutableData(output_val, (void**)&output_data);

    /* Trilinear interpolation */
    for (int64_t n = 0; n < N; n++) {
        for (int64_t d2 = 0; d2 < D2; d2++) {
            for (int64_t h2 = 0; h2 < H2; h2++) {
                for (int64_t w2 = 0; w2 < W2; w2++) {
                    int64_t grid_idx = ((n * D2 + d2) * H2 + h2) * W2 + w2;
                    float gx = grid_data[grid_idx * 3 + 0];
                    float gy = grid_data[grid_idx * 3 + 1];
                    float gz = grid_data[grid_idx * 3 + 2];

                    /* Convert from [-1,1] to [0, size-1] */
                    float fx = ((gx + 1.0f) * 0.5f) * (float)(W - 1);
                    float fy = ((gy + 1.0f) * 0.5f) * (float)(H - 1);
                    float fz = ((gz + 1.0f) * 0.5f) * (float)(D - 1);

                    int x0 = (int)floorf(fx);
                    int y0 = (int)floorf(fy);
                    int z0 = (int)floorf(fz);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    float wx = fx - (float)x0;
                    float wy = fy - (float)y0;
                    float wz = fz - (float)z0;

                    x0 = (int)clamp_f((float)x0, 0.0f, (float)(W - 1));
                    x1 = (int)clamp_f((float)x1, 0.0f, (float)(W - 1));
                    y0 = (int)clamp_f((float)y0, 0.0f, (float)(H - 1));
                    y1 = (int)clamp_f((float)y1, 0.0f, (float)(H - 1));
                    z0 = (int)clamp_f((float)z0, 0.0f, (float)(D - 1));
                    z1 = (int)clamp_f((float)z1, 0.0f, (float)(D - 1));

                    for (int64_t c = 0; c < C; c++) {
                        const float* vol = input_data + (n * C + c) * D * H * W;

                        float c000 = vol[z0 * H * W + y0 * W + x0];
                        float c001 = vol[z0 * H * W + y0 * W + x1];
                        float c010 = vol[z0 * H * W + y1 * W + x0];
                        float c011 = vol[z0 * H * W + y1 * W + x1];
                        float c100 = vol[z1 * H * W + y0 * W + x0];
                        float c101 = vol[z1 * H * W + y0 * W + x1];
                        float c110 = vol[z1 * H * W + y1 * W + x0];
                        float c111 = vol[z1 * H * W + y1 * W + x1];

                        float c00 = c000 * (1.0f - wx) + c001 * wx;
                        float c01 = c010 * (1.0f - wx) + c011 * wx;
                        float c10 = c100 * (1.0f - wx) + c101 * wx;
                        float c11 = c110 * (1.0f - wx) + c111 * wx;

                        float c0 = c00 * (1.0f - wy) + c01 * wy;
                        float c1 = c10 * (1.0f - wy) + c11 * wy;

                        float val = c0 * (1.0f - wz) + c1 * wz;

                        int64_t out_idx = ((n * C + c) * D2 + d2) * H2 * W2 + h2 * W2 + w2;
                        output_data[out_idx] = val;
                    }
                }
            }
        }
    }

    return NULL;
}

static void DestroyKernel(void* kernel_ptr) {
    free(kernel_ptr);
}

static int GetStartVersion(const OrtCustomOp* op) { return 1; }
static int GetEndVersion(const OrtCustomOp* op) { return 20; }
static const char* GetExecutionProviderType(const OrtCustomOp* op) { return NULL; }

static OrtCustomOp grid_sample_3d_op = {
    ORT_API_VERSION,
    CreateKernel,
    GetName,
    GetExecutionProviderType,
    GetInputType,
    GetInputTypeCount,
    GetOutputType,
    GetOutputTypeCount,
    Compute,
    DestroyKernel,
    GetInputCharacteristic,
    GetOutputCharacteristic,
    GetVariadicInputMinArity,
    GetVariadicInputHomogeneity,
    GetVariadicOutputMinArity,
    GetVariadicOutputHomogeneity,
    GetStartVersion,
    GetEndVersion,
    NULL, /* CreateKernelV2 */
    NULL, /* InferOutputShapeFn */
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
