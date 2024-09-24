from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Lambda

# Define KFold Cross Validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store accuracy and loss for each fold
fold_accuracies = []
fold_losses = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
    print(f"Training fold {fold+1}/{kf.get_n_splits()}")

    # Split data
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Normalize or scale the training data if required
    # Example: x_train_s = your_scaler.fit_transform(x_train)

    # Build a new model instance for each fold
    model = Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(filters=16, kernel_size=(3,3), activation='tanh', name='Conv2D_1'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(filters=32, kernel_size=(3,3), activation='softsign', name='Conv2D_2'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(filters=64, kernel_size=(3,3), activation='elu', name='Conv2D_3'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(filters=128, kernel_size=(3,3), name='Conv2D_4'),
        Lambda(tf.nn.crelu),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(filters=256, kernel_size=(3,3), activation='relu6', name='Conv2D_5'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(units=32, activation='relu'),
        BatchNormalization(),
        Dense(units=len(class_labels), activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Callbacks
    ES = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2, restore_best_weights=True, mode='max', min_delta=0)
    MP = ModelCheckpoint(filepath=f'Best_model_fold_{fold+1}.keras', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
    RP = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=2, min_lr=0.0001, factor=0.2)

    # Train the model on the current fold
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        validation_data=(x_val, y_val),
                        epochs=25,
                        callbacks=[ES, MP, RP],
                        verbose=1)

    # Evaluate the model on validation data for this fold
    val_loss, val_acc = model.evaluate(x_val, y_val)
    fold_accuracies.append(val_acc)
    fold_losses.append(val_loss)

# Print the final cross-validated results
print(f"Cross-validated accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Cross-validated loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
