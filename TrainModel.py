from dataclasses import asdict
from typing import List

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from Models.ArgumentsDataType import Arguments
from Models.BalloonDataType import ClassifiedBalloon


def train_and_validate6(
        args: Arguments,
        learning_data: List[ClassifiedBalloon],
        verification_data: List[ClassifiedBalloon],
) -> None:
    num_categories = max(args.num_backgrounds,
                         args.num_foregrounds, args.num_shapes)
    # Simulate a data set of categorical and numerical values
    # Configure simulation specifications: {feature: number of unique categories or None for numerical
    theColumnSpec = {
        'diameter': None,
        'background_color': args.num_backgrounds,
        'foreground_color': args.num_foregrounds,
        'shapes': args.num_shapes,
        'classification': num_categories,
    }

    # Creation of the dataset as pandas.DataFrame
    true_value_field = "classification"
    origDf = pd.DataFrame.from_records(
        [asdict(x.balloon) | {true_value_field: x.classification} for x in learning_data])
    origDf = pd.DataFrame.from_records(
        [asdict(x.balloon) for x in learning_data])
    trueValueArray = np.array([x.classification for x in learning_data])
    dfFullDataByColumn = [
        origDf['diameter'],
        origDf['background_color'].astype('category'),
        origDf['foreground_color'].astype('category'),
        origDf['shapes'].astype('category'),
    ]
    theDF = pd.concat(dfFullDataByColumn, axis=1)

    # inventory of the categorical features' values ( None for the numerical)
    theCatCodes = {
        theCol: (
            sorted(theDF[theCol].unique().tolist())
            if str(theDF[theCol].dtypes) == "category" else None)
        for theCol in theDF.columns}

    # batch size and timesteps
    theBatchSz, theTimeSteps = 10, 1

    # Creation of the batched tensorflow.data.Dataset
    theDS = tf.data.Dataset.from_tensor_slices(dict(theDF))
    theDS = theDS.window(size=theTimeSteps, shift=1,
                         stride=1, drop_remainder=True)
    theDS = theDS.flat_map(lambda x: tf.data.Dataset.zip(x))
    theDS = theDS.batch(batch_size=theBatchSz, drop_remainder=True)

    # extracting one batch
    theBatch = next(iter(theDS))
    tf.print(theBatch)

    # Creation of the components for the interface layer
    theFeaturesInputs = {}
    theFeaturesEncoded = {}

    for theFeature, theCodes in theCatCodes.items():
        if theCodes is None:  # Pass-through for numerical features
            theNumInput = tf.keras.layers.Input(
                shape=[], dtype=tf.float32, name=theFeature)
            theFeaturesInputs[theFeature] = theNumInput

            theFeatureExp = tf.expand_dims(input=theNumInput, axis=-1)
            theFeaturesEncoded[theFeature] = theFeatureExp

        else:  # Process for categorical features
            theCatInput = tf.keras.layers.Input(
                shape=[], dtype=tf.int64, name=theFeature)
            theFeaturesInputs[theFeature] = theCatInput

            theFeatureExp = tf.expand_dims(input=theCatInput, axis=-1)
            theEncodingLayer = tf.keras.layers.CategoryEncoding(
                num_tokens=theColumnSpec[theFeature],
                name=f"{theFeature}_enc",
                output_mode="one_hot", sparse=False)
            theFeaturesEncoded[theFeature] = theEncodingLayer(theFeatureExp)

    # Below is what you'd be interested in

    theStackedInputs = tf.concat(tf.nest.flatten(theFeaturesEncoded), axis=1)

    theModel = tf.keras.Model(inputs=theFeaturesInputs,
                              outputs=theStackedInputs)
    theOutputs = theModel(theBatch)
    assert theOutputs is not None
    tf.print(theOutputs[:5], summarize=-1)

    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(theStackedInputs)
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    theModelOutputs = tf.keras.layers.Dense(
        num_categories, activation=tf.nn.softmax)(x)

    theFullModel = tf.keras.Model(
        inputs=theFeaturesInputs, outputs=theModelOutputs)

    theFullModel.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    history = theFullModel.fit(
        x=theBatch, y=trueValueArray, epochs=10, validation_split=0.1)

    theOutput = theFullModel(theBatch)
    tf.print(theOutput, summarize=-1)
    tf.print(trueValueArray, summarize=-1)


if __name__ == "__main__":
    from Runner import main
    main()
