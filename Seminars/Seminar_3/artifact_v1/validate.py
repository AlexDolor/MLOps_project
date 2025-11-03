import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms as transforms

# Параметры
save_path = 'Seminars/Seminar_3/'
ref_path = save_path + "model.onnx"
opt_path = save_path + "model_opt.onnx"
BATCH = 4
SEED = 42

def run_inference(session, inputs):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inputs})[0]

if __name__ == "__main__":
    np.random.seed(SEED)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH,
                                            shuffle=True, num_workers=2)
    for img, label in testloader:
        break
    inputs = img.numpy()
    
    session_ref = ort.InferenceSession(ref_path)
    session_opt = ort.InferenceSession(opt_path)
    outputs_ref = run_inference(session_ref, inputs)
    outputs_opt = run_inference(session_opt, inputs)
    
    # Расхождение
    abs_diff = np.abs(outputs_ref - outputs_opt)
    max_error = abs_diff.max()
    mean_error = abs_diff.mean()
    
    print(f"Max diff: {max_error:.6f}")
    print(f"Mean diff: {mean_error:.6f}")
    assert max_error <= 1e-3, f"FAILED: max_error={max_error:.6f} > 1e-3"
    print("PASSED: Models are equivalent within tolerance 1e-3")
