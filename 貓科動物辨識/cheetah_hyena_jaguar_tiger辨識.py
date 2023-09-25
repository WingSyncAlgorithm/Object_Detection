import os
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg後端
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm  # 導入tdqm

# 設定數據集目錄
train_dir = "D:\\數據集資料夾\\訓練"
validation_dir = "D:\\數據集資料夾\\驗證"

# 初始化變數來存儲圖像和標籤
images = []
labels = []

# 處理訓練集目錄中的每個類別
for class_name in tqdm(os.listdir(train_dir), desc="Processing Classes"):
    class_dir = os.path.join(train_dir, class_name)
    if os.path.isdir(class_dir):
        for image_file in tqdm(os.listdir(class_dir), desc=f"Processing {class_name} Images"):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(class_dir, image_file)
                image = Image.open(image_path).convert("RGB")
                image = image.resize((224, 224))  # 調整圖像大小
                images.append(np.array(image))
                labels.append(class_name)

# 轉換成NumPy數組
images = np.array(images)
labels = np.array(labels)

# 將標籤編碼為數字
le = LabelEncoder()
labels = le.fit_transform(labels)

# 切分數據集為訓練集和驗證集
X_train, X_validation, y_train, y_validation = train_test_split(images, labels, test_size=0.2, random_state=42)

# 從訓練集中隨機選取一部分作為測試集
test_size = 0.2  # 設定測試集大小
random_seed = 42  # 設定隨機種子

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_seed)

# 建立和訓練一個支持向量機（SVM）模型
svm_model = SVC(kernel='linear', C=1)

# 訓練模型
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# 驗證模型
y_pred = svm_model.predict(X_test.reshape(X_test.shape[0], -1))

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 生成混淆矩陣
confusion_mtx = confusion_matrix(y_test, y_pred)

# 定義類別名稱
class_names = le.classes_

# 繪製混淆矩陣圖表
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(confusion_mtx[i, j]), horizontalalignment='center', verticalalignment='center', color='white' if i == j else 'black')

plt.ylabel('True label')
plt.xlabel('Predicted label')

# 計算學習曲線
def calculate_learning_curve(estimator, X, y, cv, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_scores = []
    validation_scores = []
    
    for train_size in tqdm(train_sizes, desc="Calculating Learning Curve"):
        train_subset_size = int(train_size * X.shape[0])
        X_subset = X[:train_subset_size]
        y_subset = y[:train_subset_size]
        
        estimator.fit(X_subset.reshape(X_subset.shape[0], -1), y_subset)
        
        train_score = estimator.score(X_subset.reshape(X_subset.shape[0], -1), y_subset)
        validation_score = estimator.score(X_test.reshape(X_test.shape[0], -1), y_test)
        
        train_scores.append(train_score)
        validation_scores.append(validation_score)
    
    return train_scores, validation_scores

# 計算學習曲線
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, validation_scores = calculate_learning_curve(svm_model, X_train, y_train, cv=None, train_sizes=train_sizes)

# 計算損失曲線
train_loss = []
validation_loss = []
for train_size in tqdm(train_sizes, desc="Calculating Loss Curve"):
    train_subset_size = int(train_size * X_train.shape[0])
    X_subset = X_train[:train_subset_size]
    y_subset = y_train[:train_subset_size]
    
    svm_model.fit(X_subset.reshape(X_subset.shape[0], -1), y_subset)
    
    train_loss.append(1 - svm_model.score(X_subset.reshape(X_subset.shape[0], -1), y_subset))
    validation_loss.append(1 - svm_model.score(X_test.reshape(X_test.shape[0], -1), y_test))

# 繪製學習曲線和損失曲線
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_sizes * X_train.shape[0], train_scores, 'r', label='Training accuracy')
plt.plot(train_sizes * X_train.shape[0], validation_scores, 'b', label='Validation accuracy')
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_sizes * X_train.shape[0], train_loss, 'r', label='Training loss')
plt.plot(train_sizes * X_train.shape[0], validation_loss, 'b', label='Validation loss')
plt.title('Loss Curve')
plt.xlabel('Training examples')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 生成分類報告
report = classification_report(y_test, y_pred, target_names=class_names)
print("Classification Report:")
print(report)
