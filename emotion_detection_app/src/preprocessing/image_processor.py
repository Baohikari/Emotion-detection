from tf_keras.models import Sequential
from tf_keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    BatchNormalization, GlobalAveragePooling2D, LeakyReLU
)
from tf_keras.optimizers.legacy import Adam
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from tf_keras.regularizers import l2

# Đường dẫn dữ liệu
project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data" / "FER-2013"
train_dir = data_dir / "train"
validation_dir = data_dir / "test"

# Tạo ImageDataGenerator
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Chia nhỏ tập train để làm validation
)
test_data_gen = ImageDataGenerator(rescale=1./255)

# Chuẩn bị dữ liệu
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # FER-2013 là 48x48
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)
validation_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Tạo mô hình
emotion_model = Sequential([
    Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=(48, 48, 1)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    GlobalAveragePooling2D(),
    Dense(512, kernel_regularizer=l2(0.01)),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(7, activation='softmax')  # FER-2013 có 7 nhãn
])

emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0003),
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
model_checkpoint = ModelCheckpoint(
    filepath=str(project_dir / "models" / "best_emotion_model.h5"),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Huấn luyện mô hình
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Độ chính xác cuối cùng
final_train_accuracy = emotion_model_info.history['accuracy'][-1]
final_val_accuracy = emotion_model_info.history['val_accuracy'][-1]
print(f"\nĐộ chính xác cuối cùng trên tập huấn luyện: {final_train_accuracy:.4f}")
print(f"Độ chính xác cuối cùng trên tập validation: {final_val_accuracy:.4f}")
