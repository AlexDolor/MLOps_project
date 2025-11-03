from model_func import MnistNet, models_differences, measure_performance_torch,\
     measure_performance_onnx
import torch
from torch.quantization import quantize_dynamic
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import onnx
import onnxoptimizer as onnx_optim
import onnxruntime as ort

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    batch_size = 4
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    metrics_df = pd.DataFrame(columns=['latency (ms)', 'throughput (obj/sec)', 'accuracy'])
    net = MnistNet()
    net.load_state_dict(torch.load('weights/MnistNet.pt', weights_only=True))
    scripted = torch.jit.script(net)
    scripted.save("weights/MnistNet_scripted.pt")
    
    metrics = measure_performance_torch(net, testloader, 10000)
    metrics_df.loc['default_model'] = metrics[:-1]

    ##quantize model 
    model_int8 = quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
    print('quant differences')
    models_differences(net, model_int8, testloader)
    metrics = measure_performance_torch(model_int8, testloader, 10000)
    metrics_df.loc['model_quant'] = metrics[:-1]
    torch.save(model_int8.state_dict(), 'weights/MnistNet_quant.pt')
    scripted = torch.jit.script(model_int8)
    scripted.save("weights/MnistNet_quant_scripted.pt")
    

    # export model to onnx
    for img, label in testloader:
        break
    net.eval()
    torch.onnx.export(
        net, img, "onnx_models/MnistNet.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13
    )
    sess = ort.InferenceSession("onnx_models/MnistNet.onnx")
    metrics = measure_performance_onnx(sess, testloader, 10000)
    metrics_df.loc['onnx_model'] = metrics[:-1]

    # optimize onnx model
    passes = ["eliminate_deadend", "fuse_add_bias_into_conv", "fuse_bn_into_conv"]
    model_optimized = onnx_optim.optimize(onnx.load("onnx_models/MnistNet.onnx"), passes)
    onnx.save(model_optimized, "onnx_models/MnistNet_opt.onnx")

    sess = ort.InferenceSession("onnx_models/MnistNet_opt.onnx")
    metrics = measure_performance_onnx(sess, testloader, 10000)
    metrics_df.loc['onnx_model_opt'] = metrics[:-1]
    
    print(metrics_df)
    metrics_df.to_csv('model_metrics.csv')
