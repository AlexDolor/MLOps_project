### План и список заданий с решениями для первого семинара


### **План семинара:**

**Установка Python и настройка виртуального окружения (30 минут)**
   - **Шаг 1:** Скачивание и установка Python.
     - Перейти на официальный сайт [Python](https://www.python.org/downloads/) и скачать последнюю стабильную версию Python 3.8 или выше.
     - Следовать инструкциям установщика (не забудьте отметить опцию "Add Python to PATH").
   - **Шаг 2:** Установка `pip` (если не установлен).
     - `pip` обычно устанавливается вместе с Python. Проверить установку командой:
       ```bash
       pip --version
       ```
   - **Шаг 3:** Установка и настройка виртуального окружения.
     - Создание виртуального окружения:
       ```bash
       python -m venv ml_env
       ```
     - Активация виртуального окружения:
       - **Windows:**
         ```bash
         ml_env\Scripts\activate
         ```
       - **macOS/Linux:**
         ```bash
         source ml_env/bin/activate
         ```
     - Проверка активированного окружения (появление префикса `(ml_env)` в командной строке).

3. **Установка необходимых библиотек и фреймворков (30 минут)**
   - **Шаг 1:** Обновление `pip`.
     ```bash
     pip install --upgrade pip
     ```
   - **Шаг 2:** Установка PyTorch.
     - Перейти на [официальный сайт PyTorch](https://pytorch.org/get-started/locally/) для выбора команды установки в зависимости от вашей ОС и наличия GPU.
     - Пример установки для CPU:
       ```bash
       pip install torch torchvision torchaudio
       ```
   - **Шаг 3:** Установка дополнительных библиотек.
     ```bash
     pip install matplotlib numpy pandas scikit-learn onnxruntime tensorboard
     ```
   - **Шаг 4:** Установка Jupyter Notebook или Visual Studio Code.
     - **Jupyter Notebook:**
       ```bash
       pip install notebook
       ```
     - **Visual Studio Code:**
       - Скачайте и установите с [официального сайта](https://code.visualstudio.com/).
       - Установите расширение Python через встроенный менеджер расширений.
   
4. **Установка Docker (если требуется для будущих лекций) (20 минут)**
   - Перейти на [официальный сайт Docker](https://www.docker.com/get-started) и скачать Docker Desktop для вашей операционной системы.
   - Следовать инструкциям установщика.
   - Проверка установки командой:
     ```bash
     docker --version
     ```
   - **Примечание:** Docker требует наличия учетной записи. Зарегистрируйтесь на [Docker Hub](https://hub.docker.com/) при необходимости.

5. **Настройка Git и GitHub (15 минут)**
   - **Шаг 1:** Установка Git.
     - Перейти на [официальный сайт Git](https://git-scm.com/downloads) и скачать установщик для вашей ОС.
     - Следовать инструкциям установщика.
   - **Шаг 2:** Настройка Git.
     ```bash
     git config --global user.name "Ваше Имя"
     git config --global user.email "ваш.email@example.com"
     ```
   - **Шаг 3:** Создание учетной записи на GitHub (если еще нет).
     - Перейти на [GitHub](https://github.com/) и зарегистрироваться.
   - **Шаг 4:** Создание SSH-ключа для безопасного взаимодействия с GitHub.
     ```bash
     ssh-keygen -t ed25519 -C "ваш.email@example.com"
     ```
     - Следовать инструкциям и добавить публичный ключ в настройки GitHub.

6. **Проверка корректности установки (15 минут)**
   - **Проверка Python и библиотек:**
     ```python
     import torch
     import torchvision
     import matplotlib
     import onnxruntime
     import tensorboard
     print("Все библиотеки успешно установлены!")
     ```
     - Запустить этот код в Jupyter Notebook или скрипте Python.
   - **Запуск TensorBoard:**
     ```bash
     tensorboard --logdir=runs
     ```
     - Открыть браузер и перейти по адресу [http://localhost:6006/](http://localhost:6006/) для проверки работоспособности.
   


### **Список заданий для семинара:**

#### **Задание 1: Установка и настройка окружения**
   
**Описание:**
Выполните все шаги по установке Python, настройке виртуального окружения, установке необходимых библиотек и инструментов, описанные в плане семинара. Убедитесь, что все компоненты работают корректно, выполнив проверочный скрипт.

**Шаги выполнения:**
1. Установите Python и настройте виртуальное окружение.
2. Установите PyTorch и дополнительные библиотеки.
3. Установите и настройте Jupyter Notebook или Visual Studio Code.
4. Установите Docker и Git, настройте GitHub.
5. Запустите проверочный скрипт для проверки установки.
6. Запустите TensorBoard и убедитесь в его работоспособности.

**Решение:**
*Пример проверочного скрипта:*

```python
import torch
import torchvision
import matplotlib.pyplot as plt
import onnxruntime as ort
import tensorboard

print("PyTorch версия:", torch.__version__)
print("Torchvision версия:", torchvision.__version__)
print("Matplotlib версия:", plt.__version__)
print("ONNX Runtime версия:", ort.__version__)
print("TensorBoard версия:", tensorboard.__version__)

# Проверка доступности GPU
if torch.cuda.is_available():
    print("CUDA доступна. Используется GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA недоступна. Используется CPU.")
```

*Ожидаемый вывод:*
```
PyTorch версия: 1.13.1
Torchvision версия: 0.14.1
Matplotlib версия: 3.5.2
ONNX Runtime версия: 1.12.1
TensorBoard версия: 2.8.0
CUDA доступна. Используется GPU: NVIDIA GeForce GTX 1080 Ti
```

#### **Задание 2: Создание и запуск простого проекта**

**Описание:**
Создайте простой проект для проверки работоспособности установленной среды. Напишите скрипт, который загружает датасет MNIST, отображает несколько изображений и запускает TensorBoard для визуализации.

**Шаги выполнения:**
1. Создайте директорию для проекта.
2. Напишите скрипт `project.py` со следующим содержанием:

```python
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Настройка трансформаций
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Загрузка датасета MNIST
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Визуализация нескольких изображений
def imshow(img):
    img = img / 2 + 0.5  # Денормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Получение одной партии данных
dataiter = iter(train_loader)
images, labels = dataiter.next()

# Отображение первых 4 изображений
imshow(torchvision.utils.make_grid(images[:4]))
print('GroundTruth: ', ' '.join(str(labels[j].item()) for j in range(4)))

# Запуск TensorBoard
writer = SummaryWriter('runs/test_run')
writer.add_images('mnist_images', images[:4], 0)
writer.close()
```

3. Запустите скрипт:
   ```bash
   python project.py
   ```
4. Запустите TensorBoard:
   ```bash
   tensorboard --logdir=runs
   ```
5. Откройте браузер и перейдите по адресу [http://localhost:6006/](http://localhost:6006/) для просмотра визуализации.

**Решение:**
*После запуска скрипта вы увидите отображение нескольких изображений из датасета MNIST и метки классов. В TensorBoard будет отображена секция с изображениями, подтверждающая корректность интеграции.*

---

### **Домашнее задание:**

1. **Задание 1:** 
   - Проверьте установку всех инструментов, выполнив проверочный скрипт из Задания 1.
   - Убедитесь, что TensorBoard запускается корректно и отображает изображения.

2. **Задание 2:**
   - Создайте собственный скрипт, который загружает другой датасет из `torchvision.datasets`, отображает несколько примеров и логирует их в TensorBoard.
   - Пример: Используйте датасет CIFAR-10 вместо MNIST.


---

### **Рекомендации для выполнения заданий:**

- **Внимательно следуйте инструкциям:** Убедитесь, что все команды вводятся правильно, особенно при установке и настройке окружения.
- **Документируйте процесс:** Ведите записи о возникающих проблемах и способах их решения. Это поможет вам и вашим коллегам в будущем.
- **Используйте официальные ресурсы:** При возникновении трудностей обращайтесь к официальной документации PyTorch, Docker, GitHub и других инструментов.
- **Общайтесь с преподавателем и однокурсниками:** Если у вас возникли вопросы или проблемы, не стесняйтесь обращаться за помощью.

---


### **Дополнительные ресурсы:**

- **Официальная документация PyTorch:**
  - [PyTorch Tutorials](https://pytorch.org/tutorials/)
  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

- **Документация TensorBoard:**
  - [TensorBoard Overview](https://www.tensorflow.org/tensorboard)

- **Онлайн-курсы:**
  - [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
  - [Fast.ai's Practical Deep Learning for Coders](https://course.fast.ai/)

- **Статьи и блоги:**
  - [Understanding Virtual Environments in Python](https://realpython.com/python-virtual-environments-a-primer/)
  - [Getting Started with TensorBoard](https://www.tensorflow.org/tensorboard/get_started)

