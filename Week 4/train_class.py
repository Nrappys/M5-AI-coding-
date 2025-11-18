import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    """
    คลาสนี้เก็บเฉพาะ "ตรรกะ" ของโมเดล (Math)
    จะไม่ยุ่งเกี่ยวกับการวนลูปเทรน, การสับข้อมูล, หรือการแบ่ง Batch
    """
    def __init__(self, layer_dims, learning_rate=0.1, lambd=0.1):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.alpha = learning_rate 
        self.lambd = lambd
        self.parameters = {}
        self.L = len(layer_dims) - 1 
        self._init_params()

    def _init_params(self):
        np.random.seed(1) 
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    # --- Activation Functions ---
    def _relu(self, Z):
        return np.maximum(Z, 0)

    def _deriv_relu(self, Z):
        return Z > 0

    def _softmax(self, Z):
        Z = Z - np.max(Z, axis=0, keepdims=True) 
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        return A
    
    def _one_hot(self, Y, m):
        one_hot_Y = np.zeros((self.layer_dims[self.L], m))
        one_hot_Y[Y, np.arange(m)] = 1
        return one_hot_Y

    # --- Core Model Logic ---
    def _forward_prop(self, X):
        caches = []
        A = X
        A_prev = X
        
        for l in range(1, self.L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            A = self._relu(Z)
            cache = (A_prev, W, b, Z)
            caches.append(cache)
            A_prev = A

        W_L = self.parameters[f'W{self.L}']
        b_L = self.parameters[f'b{self.L}']
        Z_L = np.dot(W_L, A_prev) + b_L
        AL = self._softmax(Z_L)
        cache = (A_prev, W_L, b_L, Z_L)
        caches.append(cache)
        return AL, caches

    def _compute_loss(self, AL, Y, m):
        log_probs = -np.log(AL[Y, np.arange(m)])
        data_loss = np.sum(log_probs) / m
        
        reg_loss = 0
        for l in range(1, self.L + 1):
            reg_loss += np.sum(np.square(self.parameters[f'W{l}']))
        reg_loss = (self.lambd / (2 * m)) * reg_loss
        return data_loss + reg_loss

    def _back_prop(self, AL, Y, m, caches):
        grads = {}
        one_hot_Y = self._one_hot(Y, m)
        
        dZL = AL - one_hot_Y 
        A_prev, WL, bL, ZL = caches[self.L - 1]
        
        grads[f'dW{self.L}'] = (1/m) * np.dot(dZL, A_prev.T) + (self.lambd / m) * WL
        grads[f'db{self.L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        
        dAPrev = np.dot(WL.T, dZL) 

        for l in reversed(range(1, self.L)): 
            A_prev, W, b, Z = caches[l - 1] 
            dZ = dAPrev * self._deriv_relu(Z)
            grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T) + (self.lambd / m) * W
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = np.dot(W.T, dZ)
        return grads

    def _update_params(self, grads):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= self.alpha * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.alpha * grads[f'db{l}']
            
    # --- Learning Rate Decay ---
    def update_learning_rate(self, epoch_num):
        """อัปเดต alpha ที่เก็บไว้ในคลาส"""
        self.alpha = self.learning_rate * (1 / (1 + 0.0001 * epoch_num))

    # --- Public Interface ---
    def predict(self, X):
        AL, _ = self._forward_prop(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / len(Y)
    
    # ---------------------------------------------------------------------
# ⭐️ ฟังก์ชัน train (ส่วนที่ "ยุ่ง" กับข้อมูล) จะอยู่ข้างนอกคลาส ⭐️
# ---------------------------------------------------------------------

def train_model(model, X_train, Y_train, X_val, Y_val, n_iters, batch_size):
    """
    ฟังก์ชันนี้จะรับ "โมเดล" ที่บริสุทธิ์เข้ามา
    และทำหน้าที่ "แบ่ง Batch", และ "วนลูป" การเทรนให้
    (ลบการ Shuffle ภายในลูปออกแล้ว)
    """
    m = X_train.shape[1]
    
    costs = []
    accuracies = []
    iterations = []

    print(f"Starting training for {n_iters} iterations...")
    
    start_time_iter = time.time()
    
    for i in range(n_iters + 1):
        
        # --- 1. ส่วนที่ "ยุ่ง" กับข้อมูล (Batching) ---
        # ⭐️⭐️⭐️ START: แก้ไขจุดนี้ ⭐️⭐️⭐️
        # ลบการ Shuffle ภายในลูปออก
        # permutation = np.random.permutation(m)
        # X_shuffled = X_train[:, permutation]
        # Y_shuffled = Y_train[permutation]

        for j in range(0, m, batch_size):
            # ใช้อินพุต X_train, Y_train โดยตรง
            X_batch = X_train[:, j:j+batch_size]
            Y_batch = Y_train[j:j+batch_size]
            # ⭐️⭐️⭐️ END: แก้ไขจุดนี้ ⭐️⭐️⭐️
            
            m_batch = Y_batch.size

            # --- 2. เรียกใช้ "ตรรกะ" ของโมเดล ---
            AL, caches = model._forward_prop(X_batch)
            grads = model._back_prop(AL, Y_batch, m_batch, caches)
            model._update_params(grads) # โมเดลอัปเดตพารามิเตอร์ของตัวเอง

        end_time_iter = time.time() # ⭐️ จับเวลาสิ้นสุดรอบ
        time_per_iter = end_time_iter - start_time_iter

        # --- 3. ส่วนประเมินผลและ Log (ยุ่งกับ Data) ---
        if i % 1 == 0:
            # ประเมินผลด้วย Validation data
            AL_val, _ = model._forward_prop(X_val)
            loss = model._compute_loss(AL_val, Y_val, Y_val.size)
            
            predictions = np.argmax(AL_val, axis=0) 
            accuracy = model.get_accuracy(predictions, Y_val)
            
            costs.append(loss)
            accuracies.append(accuracy)
            iterations.append(i)
            
            print(f"Iteration {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {time_per_iter:.2f}s")
            if accuracy >= 0.93:
                print("93% leaw Ja")
                break
        
        # --- 4. อัปเดต Learning Rate ในโมเดล ---
        model.update_learning_rate(i)

    # --- 5. พล็อต (ส่วนย่อยของ "กระบวนการ" เทรน) ---
    _plot_graphs(iterations, accuracies, costs)
    
    return model # คืนค่าโมเดลที่เทรนเสร็จแล้ว

def _plot_graphs(iteration_list, accuracy_list, loss_list):
    """
    ฟังก์ชันพล็อต (ย้ายออกมาจากคลาส)
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iteration_list, accuracy_list, label="Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Iterations")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iteration_list, loss_list, label="Loss", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over Iterations")
    plt.grid()
    plt.legend()
    plt.show()

# ---------------------------------------------------------------------
# ส่วนที่ 1: โหลดข้อมูล (ส่วนนี้ "ยุ่ง" กับ Data)
# ---------------------------------------------------------------------
print("Loading and preparing data...")
data = pd.read_csv(r'Data/train.csv').to_numpy()
np.random.shuffle(data) # สับเปลี่ยน (ยังคงสับครั้งแรกตอนโหลดข้อมูล)
data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:] / 255.0 # Normalization
data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:] / 255.0 # Normalization
print("Data ready.")

# ---------------------------------------------------------------------
# ส่วนที่ 2: การเรียกใช้งาน
# ---------------------------------------------------------------------
layers_config = [784, 1000, 1000, 1000, 10] 

# 1. สร้าง "โมเดล" (ที่ยังไม่ยุ่งกับข้อมูล)
model = DeepNeuralNetwork(layer_dims=layers_config, 
                          learning_rate=0.01)

# 2. ส่ง "โมเดล" และ "ข้อมูล" ไปให้ฟังก์ชัน "train_model" 
#    ฟังก์ชันนี้จะเป็นคนจัดการ "ยุ่ง" กับข้อมูลให้
trained_model = train_model(model, 
                            X_train, Y_train, 
                            X_dev, Y_dev, 
                            n_iters=3200, 
                            batch_size=64)

print("Training complete.")