# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

### STEP 2:

### STEP 3:

Write your own steps

## PROGRAM

{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/petchi11/mnist-classification/blob/main/Ex03_minist_classification.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "Ms2HU22Nmxkg"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "from tensorflow.keras.datasets import mnist\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "from tensorflow.keras import utils\n",
        "import pandas as pd\n",
        "from sklearn.metrics import classification_report,confusion_matrix\n",
        "from tensorflow.keras.preprocessing import image"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "gyyDcEJBoPWh",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5d33e9ca-be25-4dc4-fb9a-4b30398d53d6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
            "11493376/11490434 [==============================] - 0s 0us/step\n",
            "11501568/11490434 [==============================] - 0s 0us/step\n"
          ]
        }
      ],
      "source": [
        "(X_train, y_train), (X_test, y_test) = mnist.load_data()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "W0thCGmwocfQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "02d92b9e-11aa-4680-d94a-6d2d754ca684"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ],
      "source": [
        "X_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "Kl1HVshDojow",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fbdfcf4b-a873-42b9-c7da-0b9d4b8ab82d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10000, 28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ],
      "source": [
        "X_test.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "sUtPtTH8pYho"
      },
      "outputs": [],
      "source": [
        "single_image= X_train[100]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "d_7A8n_JpexA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1db206b6-4d7c-4f1c-8c2e-c7118e07958c"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ],
      "source": [
        "single_image.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "qyuxyqKZpiAY",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "34c2b82f-2b98-48d7-e6b6-c8f54fbb18e3"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7ff505794c90>"
            ]
          },
          "metadata": {},
          "execution_count": 8
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "plt.imshow(single_image,cmap='gray')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "gamIl8scp_vg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d617b86f-17a6-474e-d4c6-2418d226a4e8"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000,)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ],
      "source": [
        "y_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "p1Hr1eHcr7EB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "562ded14-ae9d-4678-d384-dbbd39957b28"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ],
      "source": [
        "X_train.min()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "TbytbmcjsFcJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "95827665-115e-4da9-d799-f2bcc9ec4950"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "255"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ],
      "source": [
        "X_train.max()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "D-L5mmALsIHR"
      },
      "outputs": [],
      "source": [
        "X_train_scaled = X_train/255.0\n",
        "X_test_scaled = X_test/255.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "O_5QWtIVsZZp",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bea68db8-0424-4720-8143-5b63032913b5"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.0"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ],
      "source": [
        "X_train_scaled.min()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "RSjbbOiYse95",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a035f1d0-b089-48ed-a955-f6a4d9a35563"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ],
      "source": [
        "X_train_scaled.max()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "DBXrOqnVqGTY",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c653fbd8-24b7-41bc-b7a3-f3f579352578"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "5"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "y_train[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "oL7Pld1Qrd5x"
      },
      "outputs": [],
      "source": [
        "y_train_onehot = utils.to_categorical(y_train,10)\n",
        "y_test_onehot = utils.to_categorical(y_test,10)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "ZN9h128GrH_5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "22d8ac6f-6c8a-4061-ad46-766093a45f52"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "numpy.ndarray"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ],
      "source": [
        "type(y_train_onehot)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "BTaP6Ynlrp9p",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "56359beb-9801-4f36-e2ab-285f9fbdefcf"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 10)"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ],
      "source": [
        "y_train_onehot.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "KJVyMJOSQpQi",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "d172907b-28c9-4b87-ec84-c77f80790eca"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7ff505291510>"
            ]
          },
          "metadata": {},
          "execution_count": 19
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "single_image = X_train[500]\n",
        "plt.imshow(single_image,cmap='gray')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "ppoll2_iQY57",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0f92fe4d-9799-4d41-f1ed-506eba87792e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ],
      "source": [
        "y_train_onehot[500]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "-6H82O2ouNRq"
      },
      "outputs": [],
      "source": [
        "X_train_scaled = X_train_scaled.reshape(-1,28,28,1)\n",
        "X_test_scaled = X_test_scaled.reshape(-1,28,28,1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "cXIbBlbasjaZ"
      },
      "outputs": [],
      "source": [
        "model = keras.Sequential()\n",
        "model.add(layers.Input(shape=(28,28,1)))\n",
        "model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))\n",
        "model.add(layers.MaxPool2D(pool_size=(2,2)))\n",
        "model.add(layers.Flatten())\n",
        "model.add(layers.Dense(32,activation='relu'))\n",
        "model.add(layers.Dense(10,activation='softmax'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "H5g5Ek6CgssX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "da63e958-97dd-480c-a40b-e27ef4643081"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d (Conv2D)             (None, 26, 26, 32)        320       \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         \n",
            " )                                                               \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 5408)              0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 32)                173088    \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 10)                330       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 173,738\n",
            "Trainable params: 173,738\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "tx9Sw_xqHtqI"
      },
      "outputs": [],
      "source": [
        "# Choose the appropriate parameters\n",
        "model.compile(loss='categorical_crossentropy',\n",
        "              optimizer='adam',\n",
        "              metrics='accuracy')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "oO6tpvb5Ii14",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f025575d-4b4a-4b62-c3f3-df1e6019a745"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "938/938 [==============================] - 27s 28ms/step - loss: 0.2458 - accuracy: 0.9289 - val_loss: 0.0910 - val_accuracy: 0.9734\n",
            "Epoch 2/5\n",
            "938/938 [==============================] - 25s 27ms/step - loss: 0.0816 - accuracy: 0.9768 - val_loss: 0.0662 - val_accuracy: 0.9780\n",
            "Epoch 3/5\n",
            "938/938 [==============================] - 25s 27ms/step - loss: 0.0596 - accuracy: 0.9821 - val_loss: 0.0545 - val_accuracy: 0.9826\n",
            "Epoch 4/5\n",
            "938/938 [==============================] - 27s 28ms/step - loss: 0.0465 - accuracy: 0.9856 - val_loss: 0.0563 - val_accuracy: 0.9803\n",
            "Epoch 5/5\n",
            "938/938 [==============================] - 25s 27ms/step - loss: 0.0385 - accuracy: 0.9885 - val_loss: 0.0563 - val_accuracy: 0.9816\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7ff5013cedd0>"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ],
      "source": [
        "model.fit(X_train_scaled ,y_train_onehot, epochs=5,\n",
        "          batch_size=64, \n",
        "          validation_data=(X_test_scaled,y_test_onehot))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "id": "baRgwlwaLCqp"
      },
      "outputs": [],
      "source": [
        "metrics = pd.DataFrame(model.history.history)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "yBCYG9r9LKsp",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "51cb8d0e-55af-41d0-a5f1-21a3629091a5"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       loss  accuracy  val_loss  val_accuracy\n",
              "0  0.245837  0.928933  0.091014        0.9734\n",
              "1  0.081567  0.976817  0.066225        0.9780\n",
              "2  0.059638  0.982117  0.054469        0.9826\n",
              "3  0.046493  0.985583  0.056280        0.9803\n",
              "4  0.038486  0.988500  0.056321        0.9816"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d22e23e1-a993-404a-81ef-93f9e0457fb5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>loss</th>\n",
              "      <th>accuracy</th>\n",
              "      <th>val_loss</th>\n",
              "      <th>val_accuracy</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.245837</td>\n",
              "      <td>0.928933</td>\n",
              "      <td>0.091014</td>\n",
              "      <td>0.9734</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.081567</td>\n",
              "      <td>0.976817</td>\n",
              "      <td>0.066225</td>\n",
              "      <td>0.9780</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.059638</td>\n",
              "      <td>0.982117</td>\n",
              "      <td>0.054469</td>\n",
              "      <td>0.9826</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.046493</td>\n",
              "      <td>0.985583</td>\n",
              "      <td>0.056280</td>\n",
              "      <td>0.9803</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.038486</td>\n",
              "      <td>0.988500</td>\n",
              "      <td>0.056321</td>\n",
              "      <td>0.9816</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d22e23e1-a993-404a-81ef-93f9e0457fb5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-d22e23e1-a993-404a-81ef-93f9e0457fb5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-d22e23e1-a993-404a-81ef-93f9e0457fb5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ],
      "source": [
        "metrics.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "4Sg3ECV6LMf5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "9d6d26ed-8017-48cb-dffd-40aed5baf274"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7ff501362a90>"
            ]
          },
          "metadata": {},
          "execution_count": 28
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "metrics[['accuracy','val_accuracy']].plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "id": "A906k0lmLOgg",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "270cb260-7a0c-4aba-a547-938a95ff6c58"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7ff501299290>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "metrics[['loss','val_loss']].plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "id": "PDnRigNeLk7B"
      },
      "outputs": [],
      "source": [
        "x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "gP5Ud8DbLpvI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bc8e6ee9-bb36-42cf-d551-f8737416fe53"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 972    0    1    0    1    1    1    1    1    2]\n",
            " [   0 1132    2    1    0    0    0    0    0    0]\n",
            " [   3    4 1012    1    2    0    0    6    2    2]\n",
            " [   0    0    4  992    0    8    0    3    2    1]\n",
            " [   0    0    0    0  975    0    0    0    0    7]\n",
            " [   2    0    1    4    0  882    2    0    1    0]\n",
            " [  10    3    1    0    6    4  932    0    2    0]\n",
            " [   0    4   10    2    2    0    0 1007    1    2]\n",
            " [   5    0    8    4    3    3    3    6  921   21]\n",
            " [   0    2    0    1    9    4    0    2    0  991]]\n"
          ]
        }
      ],
      "source": [
        "print(confusion_matrix(y_test,x_test_predictions))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "id": "9gJ7WV95L7my",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "04182dc7-b464-4077-d736-80b0ea575260"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.98      0.99      0.99       980\n",
            "           1       0.99      1.00      0.99      1135\n",
            "           2       0.97      0.98      0.98      1032\n",
            "           3       0.99      0.98      0.98      1010\n",
            "           4       0.98      0.99      0.98       982\n",
            "           5       0.98      0.99      0.98       892\n",
            "           6       0.99      0.97      0.98       958\n",
            "           7       0.98      0.98      0.98      1028\n",
            "           8       0.99      0.95      0.97       974\n",
            "           9       0.97      0.98      0.97      1009\n",
            "\n",
            "    accuracy                           0.98     10000\n",
            "   macro avg       0.98      0.98      0.98     10000\n",
            "weighted avg       0.98      0.98      0.98     10000\n",
            "\n"
          ]
        }
      ],
      "source": [
        "print(classification_report(y_test,x_test_predictions))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KlBK9Iw_MHc0"
      },
      "source": [
        "\n",
        "\n",
        "```\n",
        "# This is formatted as code\n",
        "```\n",
        "\n",
        "**Prediction for a single input**\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "mPYlnjziPPKY"
      },
      "outputs": [],
      "source": [
        "img = image.load_img('five.jpg')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {
        "id": "9NlIpMcgPQS5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2eeb000f-679c-4bc6-c551-13dce9ac4081"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "PIL.JpegImagePlugin.JpegImageFile"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ],
      "source": [
        "type(img)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "id": "Gho9nRGPMOO9"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.preprocessing import image\n",
        "img = image.load_img('five.jpg')\n",
        "img_tensor = tf.convert_to_tensor(np.asarray(img))\n",
        "img_28 = tf.image.resize(img_tensor,(28,28))\n",
        "img_28_gray = tf.image.rgb_to_grayscale(img_28)\n",
        "img_28_gray_scaled = img_28_gray.numpy()/255.0\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "id": "yrw9d6T8OXLh"
      },
      "outputs": [],
      "source": [
        "x_single_prediction = np.argmax(\n",
        "    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),\n",
        "     axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "metadata": {
        "id": "J5YWILZSPgnJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "61dbaa1f-b801-45ed-a85e-96d45e8cf376"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[5]\n"
          ]
        }
      ],
      "source": [
        "print(x_single_prediction)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "metadata": {
        "id": "P0De-3CVPpXZ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "outputId": "e04d78cd-b05e-474a-a0fb-6cfc5d06edd7"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7ff5011e3ed0>"
            ]
          },
          "metadata": {},
          "execution_count": 38
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 39,
      "metadata": {
        "id": "qqh74INOfnjX"
      },
      "outputs": [],
      "source": [
        "img_28_gray_inverted = 255.0-img_28_gray\n",
        "img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 40,
      "metadata": {
        "id": "08peSjZ2f6xG"
      },
      "outputs": [],
      "source": [
        "x_single_prediction = np.argmax(\n",
        "    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),\n",
        "     axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 41,
      "metadata": {
        "id": "jqoeXU7kf9Km",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b77861b9-9e77-42a7-ac04-b20116798238"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[5]\n"
          ]
        }
      ],
      "source": [
        "print(x_single_prediction)"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here

### New Sample Data Prediction

Include your sample input and output for your hand written images.

## RESULT
