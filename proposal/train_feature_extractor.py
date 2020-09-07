import keras
import video_data_generator
import models
import configuration as cfg

videogen_train = video_data_generator.VideoFrameGenerator("../ucf101/train", batch_size=16)
videogen_test = video_data_generator.VideoFrameGenerator("../ucf101/test", batch_size=16)
model = models.recurrent_feats_model()

opt = keras.optimizers.Adam(lr=1e-5, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=[
                  keras.metrics.categorical_accuracy,
                  keras.metrics.top_k_categorical_accuracy
              ])

model.fit_generator(videogen_train, epochs = 500, validation_data=videogen_test,
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(
                            filepath="trained_models/rec_feats_weights.{epoch:03d}.h5",
                            save_best_only=True,
                            monitor="val_categorical_accuracy",
                            period=20
                        ),
                        keras.callbacks.CSVLogger(
                            filename="train_history.csv"
                        )
                    ])

model.save(cfg.extractor_model_weights)
