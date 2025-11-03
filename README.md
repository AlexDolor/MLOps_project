### MLOps course notes

My notes on the course «Exploitation of ML models» (or MLOps for short)

Course contains:
- Lecture presentations
- Seminars
- Project assingment (most interesting part)

### Project
Project aims to familiarize the student with orchestrating containers.
The project contains 3 services in Docker containers orchestrated with Docker Compose:
- Preprocessing microservice
- Triton server with 4 versions of the same ML model
- Postprocessing microservice

ML model is a simple convolutional NN which is trained on MNIST dataset. 
Then, 4 versions of the model are loaded into Triron:
- base model, scripted with pytorch methods
- quatized model, scripted with pytorch methods
- model in onnx format
- model in onnx format, optimized with onnx methods
Model takes an array, representing an image and returns logits of classes

Preprocess service takes an image and converts it into array, sends to the choosen model in triton, recieves logits, 
sends them to the postprocessing service, gets predicted class and returns class index.

Postprocess service takes logits of classes and takes the index of the biggest value. 
Returns the value (predicted class)

---
