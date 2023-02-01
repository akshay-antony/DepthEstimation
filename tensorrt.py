import torch
from mobile_net import Mobile_Net_Unet
import torch_tensorrt
import numpy as np
import time


def benchmark(model, nruns=1000):
    with torch.no_grad():
        print("Warm Up")
        for _ in range(100):
            _ = model(random_input)
        total_time = 0
        for i in range(nruns):
            start_time = time.time()
            _ = model(random_input)
            time_taken = time.time() - start_time
            total_time += time_taken
            if i != 0 and i% 100 == 0:
                print(f"{i}: {total_time/(i+1)}")

if __name__ == "__main__":
    model = Mobile_Net_Unet().eval().cuda()
    inputs_32 = [torch_tensorrt.Input((1, 3, 192, 224), dtype=torch.float32)]
    inputs_16 = [torch_tensorrt.Input((1, 3, 192, 224), dtype=torch.half)]
    trt_model_fp_32 = torch_tensorrt.compile(model, 
                                             inputs=inputs_32, 
                                             enabled_precisions={torch.float32},
                                             require_full_compilation=False)
    # trt_model_fp_16 = torch_tensorrt.compile(model,
    #                                          inputs=inputs_16,
    #                                         #  truncate_long_and_double=True,
    #                                          enabled_precisions={torch.half, torch.float32})
    trt_model_fp_32.save('mobilenet_32.ts')
    # torch.jit.save(trt_model_fp_32, './mobilenet_32.ts')
    trt_model_fp_32 = torch.load('./mobilenet_32.ts')
    # trt_model_fp_16.save('mobilenet_16.ts')
    random_input = torch.randn((1, 3, 192, 224), dtype=torch.float32).cuda()
    print("For original model")
    benchmark(model)
    print("For tensorrt 32")
    trt_model_fp_32(random_input)
    benchmark(trt_model_fp_32)

    # print("For tensorrt 16")
    # benchmark(trt_model_fp_16)
    