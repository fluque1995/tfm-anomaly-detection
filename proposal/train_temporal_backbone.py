import keras
import features_data_generator
import models

videogen_train = features_data_generator.FeaturesGenerator("../img_features/train", nfeats=25, batch_size=64)
videogen_test = features_data_generator.FeaturesGenerator("../img_features/test", nfeats=25, batch_size=64)
model = models.recurrent_backbone()

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=[
                  keras.metrics.categorical_accuracy,
                  keras.metrics.top_k_categorical_accuracy
              ])

model.fit_generator(videogen_train, epochs = 1000, validation_data=videogen_test,
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(
                            filepath="trained_models/temporal_weights.{epoch:03d}.h5",
                            save_best_only=True,
                            period=50
                        )
                    ])

model.save("trained_models/temporal_weights.h5")
