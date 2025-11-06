import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_finetune_model(num_classes, input_shape=(224,224,3), base_trainable=False):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model


def main(data_dir, out_path='models/cnn_model.h5', batch_size=32, epochs=15):
    # Use ImageDataGenerator with validation split
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 rotation_range=20, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)
    train_gen = datagen.flow_from_directory(data_dir, target_size=(224,224), batch_size=batch_size, subset='training')
    val_gen = datagen.flow_from_directory(data_dir, target_size=(224,224), batch_size=batch_size, subset='validation')

    num_classes = train_gen.num_classes
    model = build_finetune_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(out_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    model.save(out_path)
    print('Saved', out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    main(args.data, epochs=args.epochs)