<!-- Project Setup Guide
Requirements
Make sure you have the following installed in your system:

Python 3.10
pip
Installation
Clone the repository to your local machine.

Install the required packages using the following command:

console
Copy code
pip install -r requirements.txt
Download the pre-trained model weights by running the following command:


gdown --fuzzy https://drive.google.com/file/d/1--2UaKVkbDD5FIMBpNaP6q4gx_WQ2TUe/view?usp=sharing
Set the environment variable SM_FRAMEWORK to tf.keras by running the following command on any one terminal:


export SM_FRAMEWORK=tf.keras
Usage
Start the backend server by running the following command on your terminal:


uvicorn app:app
Once the server is up and running, you can access the DeepLabV3 image segmentation API by sending a POST request to http://localhost:8000/unet with an image file in the body of the request.

The server will return the segmentation map of the input image as a PNG image. -->

# DeepLabV3 Image Segmentation API

## Project Setup Guide

### Requirements

Make sure you have the following installed in your system:

- Python 3.10
- pip

### Installation

1. Install the required packages using the following command:

   ```console
   pip install -r requirements.txt
   ```
<!--pip install gdown-->

2. Install gdown using the following command:

   ```console
   pip install gdown
   ```

3. Download the pre-trained model weights by running the following command:

   ```console
    gdown --fuzzy https://drive.google.com/file/d/1--2UaKVkbDD5FIMBpNaP6q4gx_WQ2TUe/view?usp=sharing
    ```

4. Set the environment variable `SM_FRAMEWORK` to `tf.keras` by running the following command on any one terminal:

    ```console
    export SM_FRAMEWORK=tf.keras
    ```

### Usage

1. Start the backend server by running the following command on your terminal:

    ```console
    uvicorn app:app
    ```

2. Once the server is up and running, you can access the DeepLabV3 image segmentation API by sending a POST request to `http://localhost:8000/unet` with an image file in the body of the request.

3. The server will return the segmentation map of the input image as a PNG image.

## API Documentation

To access the API documentation, open your web browser and go to http://127.0.0.1:8000/docs. This will take you to the Swagger UI page where you can interact with the API endpoints and view their documentation.

### Request

`POST /unet`

#### Headers

| Name | Type | Description |

| --- | --- | --- |

| Content-Type | string | `multipart/form-data` |

#### Body

| Name | Type | Description |

| --- | --- | --- |

| image | file | The image file to be segmented |

### Response

| Name | Type | Description |

| --- | --- | --- |

| image | file | The segmented image |

