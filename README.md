## Title

**Real-time Object Dimension Detection and Image Processing using YOLOv5, OpenCV, Azure Custom Vision, BLOB Storage, MySQL, and Flask**

## Overview

This project creates a real-time system for object detection and dimension measurement using YOLOv5, OpenCV, Azure Custom Vision, and Azure BLOB Storage. With MySQL for data storage and Flask for a user-friendly interface, the system processes images and videos to detect objects, measure their dimensions, and calculate distances accurately. This solution is ideal for applications like quality control and inventory management, enhancing operational efficiency and precision.

## Features

- **Image Upload and Processing**:
  - Upload images from local storage.
  - Receive processed images based on selected mode (size or distance).
  - Processed images are stored in **Azure BLOB Storage** for better data security and persistence.
  - Custom trained Azure AI model **(Custom Vision AI model)** identifies objects in the image and displays image accuracy.
- **Video Processing**:

  - Use live webcam or IPWebcam(app) to generate a live URL.
  - URL is used to identify objects using a Machine Learning model **YOLOv5**.
  - Video is processed frame by frame based on selected mode (size or distance).

- **Data Storage**:

  - Store processed image URL and filename in **Azure MySQL Database** for consistency and ease of access.

- **Model Training**:

  - Custom Vision model is trained with multiple images to ensure accuracy and robustness.

- **Web Application**:
  - Developed with **Flask**.
  - Deployed on **Azure Web App** for global access.

## Project Demo Video URL

- Please use a headset for better audio clarity.

[Click here to open Demo video](https://www.youtube.com/watch?v=b9HtieAq8EA)

## Project File Structure

- `Calculation-Of-Object-Dimension/`
  - `images/` - Contains images stored locally after processing.
  - `MachineLearningYoloModel.py` - Main program with **YOLOv5** model.
  - `README.md` - Project overview with installation guidelines.
  - `detect_object_size.py` - Python file to detect object dimensions.
  - `models/` - Directory containing machine learning models.
    - `yolov5m.pt` - YOLOv5 medium model.
    - `yolov5s.pt` - YOLOv5 small model.
  - `main.py` - Main file for object detection and distance calculation.
  - `Flask-Web-APP/` - Flask application files using Azure services.
    - `Images/` - Images for Azure Custom Vision training and tests.
    - `Test/` - Images for testing Custom Vision AI.
    - `Train/` - Images for training Custom Vision AI.
    - `static/` - Static files such as locally stored images.
    - `uploads/` - Locally stored images.
    - `templates/` - HTML files for Flask routes.
      - `index.html`
    - `app.py` - Main Flask application file.
    - `requirements.txt` - Python dependencies for Flask application.
    - `startup.txt` - Startup command for Flask application on Azure Web App.

## Installation

1. Create a virtual environment:

   ```cmd
   python -m venv env
   ```

2. Change to the virtual environment directory:

   ```cmd
   cd env
   ```

3. Clone the repository:

   ```cmd
   git clone https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/tree/main
   ```

4. Go to the project directory:

   ```cmd
   cd Calculation-Of-Object-Dimension
   ```

5. Install the required dependencies:

   ```cmd
   pip install -r requirements.txt
   ```

6. Run the main project with the YOLOv5 model:

   ```cmd
   python MachineLearningYoloModel.py
   ```

7. Run the main project without the machine learning model:

   ```cmd
   python main.py
   ```

8. Run the Flask Web App:
   ```cmd
   cd Flask-Web-APP
   pip install -r requirements.txt
   python app.py
   ```

## Azure Custom Vision API and Usage

This project uses **Azure Custom Vision Service** for image prediction. Set up and use the Custom Vision API as follows:

```python
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

ENDPOINT = "https://<your-custom-vision-endpoint>.cognitiveservices.azure.com/customvision/v3.0/Training"
PREDICTION_KEY = "<Your_Prediction_key>"
PROJECT_ID = "<Your_Project_ID>"
PUBLISHED_NAME = "<Your_Published_Name>"

credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
custom_vision_client = CustomVisionPredictionClient(ENDPOINT, credentials)
```

## Azure MySQL API Implementation and Usage

This project uses **Azure MySQL Service** to store image URLs stored in **Azure BLOB Storage**. Set up and use the MySQL API as follows:

```python
import MySQLdb

# Database configuration
app.config['MYSQL_HOST'] = 'object-detection.mysql.database.azure.com'
app.config['MYSQL_USER'] = 'objectdimension'
app.config['MYSQL_PASSWORD'] = 'Chetan@2003'
app.config['MYSQL_DB'] = 'image_processing_db'

# Initialize MySQL
mysql = MySQLdb.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    db=app.config['MYSQL_DB']
)

def save_image_to_db(filename, url):
    cursor = mysql.cursor()
    cursor.execute("INSERT INTO images (filename, url) VALUES (%s, %s)", (filename, url))
    mysql.commit()
    cursor.close()
```

## Azure Blob Storage

Images uploaded to the application are processed and stored in **Azure Blob Storage**. Set up and use Blob Storage as follows:

```python
from azure.storage.blob import BlobServiceClient

# Azure Blob Storage configuration
blob_service_client = BlobServiceClient.from_connection_string("<Include_BLOB_String>")
container_client = blob_service_client.get_container_client("<Include_BLOB_Container_Name>")

def upload_image_to_blob(image_path, image_name):
    unique_name = str(uuid.uuid4()) + "_" + image_name
    blob_client = container_client.get_blob_client(unique_name)
    with open(image_path, "rb") as data:
        blob_client.upload_blob(data)
    return blob_client.url
```

## Deployment

This application is deployed as an Azure Web App. Follow these steps:

### Step 1: Create a New Azure Web App

1. Navigate to the Azure Portal.
2. Create a new Web App by selecting "Create a resource" and then "Web App".
3. Choose the appropriate subscription, resource group, and provide a unique name for your Web App.
4. Select the appropriate runtime stack (Python 3.x).

### Step 2: Configure the Web App to Pull from Your GitHub Repository

1. In the Azure Portal, navigate to your Web App and select "Deployment Center".
2. Choose GitHub as the source control and authenticate with your GitHub account.
3. Select the repository and branch you want to deploy from.

### Step 3: Set Up Environment Variables

1. Navigate to "Configuration" under your Web App settings.
2. Add the following environment variables:

```plaintext
AZURE_STORAGE_CONNECTION_STRING=<your_azure_blob_storage_connection_string>
CUSTOM_VISION_ENDPOINT=<your_azure_custom_vision_endpoint>
CUSTOM_VISION_PREDICTION_KEY=<your_azure_custom_vision_prediction_key>
CUSTOM_VISION_PROJECT_ID=<your_azure_custom_vision_project_id>
CUSTOM_VISION_PUBLISHED_NAME=<your_custom_vision_published_name>
MYSQL_HOST=<your_mysql_host>
MYSQL_USER=<your_mysql_user>
MYSQL_PASSWORD=<your_mysql_password>
MYSQL_DB=<your_mysql_database_name>
```

### Step 4: Deploy the Application

1. Once the environment variables are set, go back to "Deployment Center" and click "Sync" to start the deployment process.
2. Monitor the deployment logs to ensure there are no errors.

## Additional Considerations

1. **Database Configuration**:

   - Ensure your MySQL database is accessible from the Azure Web App. You might need to configure the firewall rules to allow the Web App's IP address.
   - Alternatively, consider using Azure Database for MySQL for better integration with Azure services.

2. **Blob Storage Access**:

   - Ensure the Azure Web App has the necessary permissions to access the Azure Blob Storage. You might need to configure the access keys or use Managed Identity for enhanced security.

3. **Virtual Environment and Dependencies**:
   - Ensure your repository includes a `

requirements.txt` file listing all the dependencies. The Azure Web App will use this file to install the necessary packages.

4. **Static Files and Uploads**:

   - Configure the Azure Web App to handle static files and uploads correctly. You might need to adjust the `web.config` or use a storage solution like Azure Blob Storage to serve static files.

5. **Logging and Monitoring**:
   - Enable Azure Monitor to track the performance and logs of your Web App. This helps in diagnosing issues and monitoring the health of your application.

## Conclusion

By following these steps and considerations, you can successfully deploy your application to Azure Web App, ensuring it is configured correctly to leverage the cloud services efficiently.

## Final Deployed Application

- **Access the deployed application at:** https://object-dimension-detection.azurewebsites.net/

## Output

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/top.png)

## Project Contributors

- **Sowjanya H R**: Implemented Object Distance Calculation Method and helped in Implementing YOLO Machine Learning Model.
- **Sneha A**: Implemented real-time video processing for Object Dimension Detection and Object Distance Calculation.
- **Naveen S**: Helped in training the Azure Custom Vision model with custom images for better accuracy and implemented dual mode (size and distance) for the core application.

## My Contributions

- **Chetan B R**: Initiated the project, implemented code ideas from contributors, fixed various bugs, and errors. Implemented three Azure Core services: **Azure BLOB Storage**, **Azure MySQL Database**, **Azure Web App**, and AI services such as **Azure Custom Vision model**. Developed the Flask app for better front-end access and ease of deployment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This project would not have been possible without the support, guidance, and resources provided by several individuals and organizations. I would like to express my gratitude to them.

### Guidance and Mentorship

- **Future Ready Talent support Team**

### Technical Support

- **OpenAI**: For providing access to the GPT-4 model, which assisted in generating ideas and refining the project's approach.
- **Ultralytics**: For developing and maintaining the YOLOv5 model, a cornerstone of this project's object detection functionality.
- **Microsoft Azure**: For offering reliable and scalable cloud services, including Azure Custom Vision and Azure Blob Storage, essential for the project's deployment and data management.
- **Flask Community**: For developing and supporting Flask, making web application development straightforward and efficient.

### Resources and Documentation

- **OpenCV Community**: For providing extensive documentation and tutorials on computer vision techniques, crucial for the dimension measurement functionalities.
- **MySQL Community**: For creating a robust and reliable database management system that facilitated efficient data storage and retrieval.
- **OpenAI**: For resolving conflicts and major bugs.

### Contributors

- **Sowjanya H R, Sneha A, Naveen S**: For their unwavering support towards this project.

### Final Thanks

I would like to thank everyone who directly or indirectly contributed to the successful completion of this project. Your support and contributions have been invaluable, and I am truly grateful for your help and encouragement.

Thank you all.

**Chetan B R**
@CHIRANJEEVICHETAN

**Self and Contributors Emails**

`CHETAN B R` : chiranjeevichetan1996@gmail.com | @CHIRANJEEVICHETAN <br>
`SOWJANYA H R` : sonuradha988@gmail.com | @Sowjanya-HR <br>
`SNEHA A` : snehaammu087@gmail.com | @Sneha-A-2003 <br>
`NAVEEN S` : naveens1753@gmail.com | @Naveens20033 <br>

## Important Note:

- Due to Technical issues in Azure Web App I wasn't able to deploy the full functional flask app. The issue was with the OpenCV module where the complete version with GUI components was not present on Azure Web App for installation. Hence I minified my flask app and deployed a basic flask app with **Azure BLOB** and **Azure MySQL services** also **Azure Custom vision** is functional. <br>
  You can verify my project by locally installing the project and running the local server. Also the project can be verified through screenshots.

## Screenshot Related to Project

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/image-1.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/image-2.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/image-3.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/image-4.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/image-5.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-1.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-2.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-3.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-4.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-5.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-6.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-7.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-8.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-9.png)

![image](https://github.com/CHIRANJEEVICHETAN/Calculation-Of-Object-Dimension/blob/main/images/samples/object-10.png)
